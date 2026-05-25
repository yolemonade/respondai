#!/usr/bin/env python3
"""E2E: 게임 시작 → 키보드 입력 → Enter/AI 자동 이어짐 → 게임오버."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"
START_TIMEOUT = 45
NOTE_WAIT = 2
TURN_WAIT = 12


def wait_server(proc: subprocess.Popen, timeout: float) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"app exited early code={proc.returncode}")
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"server not up at {URL} within {timeout}s")


def run_e2e() -> list[str]:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py")],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_server(proc, START_TIMEOUT)
        print(f"[e2e] server ready {URL}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(1200)

            # S1 → S2
            page.get_by_role("button", name=re.compile("피아노 모드")).click()
            page.wait_for_timeout(800)
            if not page.get_by_text(re.compile(r"Round\s+1")).is_visible():
                errors.append("S2: Round 1 not visible after piano start")

            # S2 → S3
            page.get_by_role("button", name=re.compile("시작")).click()
            page.wait_for_timeout(1200)
            if not page.get_by_text("당신의 차례").is_visible():
                errors.append("S3: '당신의 차례' not visible after round start")

            total_exchanges = 5 * 3  # TOTAL_ROUNDS * MAX_EXCHANGES

            for ex_i in range(total_exchanges):
                round_num = ex_i // 3 + 1
                ex_in_round = ex_i % 3 + 1
                print(f"[e2e] exchange {ex_i + 1}/{total_exchanges} (R{round_num} E{ex_in_round})")

                # keyboard notes (bridge)
                page.keyboard.press("a")
                page.wait_for_timeout(200)
                page.keyboard.press("s")
                page.wait_for_timeout(400)

                has_notes = page.evaluate(
                    "() => /[1-9]\\/16/.test(document.body.innerText)"
                )
                if not has_notes:
                    page.wait_for_timeout(NOTE_WAIT * 1000)
                    has_notes = page.evaluate(
                        "() => /[1-9]\\/16/.test(document.body.innerText)"
                    )
                if not has_notes:
                    errors.append(
                        f"R{round_num} E{ex_in_round}: keys a,s did not add notes"
                    )

                # confirm via Enter only (no mouse)
                page.keyboard.press("Enter")
                page.wait_for_timeout(300)

                if page.get_by_text("AI 응답 중").is_visible():
                    pass
                else:
                    errors.append(f"R{round_num} E{ex_in_round}: expected 'AI 응답 중' after Enter")

                # auto-continue: must return to user turn OR round result without clicking continue
                try:
                    page.wait_for_function(
                        """() => {
                          const t = document.body.innerText;
                          return t.includes('당신의 차례') || t.includes('Round') && t.includes('결과');
                        }""",
                        timeout=TURN_WAIT * 1000,
                    )
                except Exception:
                    errors.append(
                        f"R{round_num} E{ex_in_round}: auto-continue timeout ({TURN_WAIT}s)"
                    )
                    break

                page.wait_for_timeout(400)
                body = page.inner_text("body")
                if "결과" in body and "Round" in body and ex_in_round == 3:
                    # end of round — click next (only between rounds, not mid-exchange)
                    page.get_by_role("button", name=re.compile("다음 라운드")).click()
                    page.wait_for_timeout(1000)
                    if round_num >= 5:
                        break
                    # S2 again
                    page.get_by_role("button", name=re.compile("시작")).click()
                    page.wait_for_timeout(1000)
                    if not page.get_by_text("당신의 차례").is_visible():
                        errors.append(f"after R{round_num}: S3 user turn not restored")

            # R5 끝 — S5로
            if "다시하기" not in page.inner_text("body"):
                page.get_by_role("button", name=re.compile("다음 라운드")).click()
                page.wait_for_timeout(2000)

            body = page.inner_text("body")
            if "다시하기" not in body:
                errors.append(f"S5: missing 다시하기 (snippet: {body[:350]!r})")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return errors


def main() -> int:
    print("[e2e] starting full game flow test …")
    try:
        errors = run_e2e()
    except Exception as e:
        print(f"[e2e] FATAL: {e}")
        return 1

    if errors:
        print("[e2e] FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[e2e] PASSED — start → keyboard → auto AI flow → game over")
    return 0


if __name__ == "__main__":
    sys.exit(main())
