#!/usr/bin/env python3
"""회귀: 입력 잠금 / j·k 건반 / 3번째 교환 AI 응답."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"
SERVER_READY_TIMEOUT = 45
LOCK_CHECK_TIMEOUT_MS = 20000
ROUND_TRIP_TIMEOUT_MS = 35000
ROUND_RESULT_TIMEOUT_MS = 45000


def wait_server(proc: subprocess.Popen, timeout: float = SERVER_READY_TIMEOUT) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"app died code={proc.returncode}")
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server not ready")


def start_game(page):
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(1200)
    page.get_by_role("button", name=re.compile("피아노")).click()
    page.wait_for_timeout(600)
    page.get_by_role("button", name=re.compile("^▶ 시작|시작")).click()
    page.wait_for_timeout(1200)
    assert page.get_by_text("당신의 차례").is_visible(), "S3 not shown"


def test_keyboard_j_not_k(page) -> str | None:
    """j→B4, k는 노트 추가 안 됨(화면은 C4~B4만)."""
    page.keyboard.press("j")
    page.wait_for_timeout(350)
    body_j = page.inner_text("body")
    page.keyboard.press("Backspace")
    page.wait_for_timeout(300)
    page.keyboard.press("k")
    page.wait_for_timeout(350)
    body_k = page.inner_text("body")
    has_b4 = "B4" in body_j or "/16" in body_j
    k_added = body_k.count("/16") > body_j.count("/16") - 1 if "/16" in body_j else "/16" in body_k
    if not has_b4:
        return f"j did not add B4 note, body={body_j[:200]!r}"
    if k_added:
        return f"k incorrectly added note (should be ignored), j={body_j[:120]!r} k={body_k[:120]!r}"
    return None


def test_enter_locked_during_ai(page) -> str | None:
    page.keyboard.press("a")
    page.wait_for_timeout(200)
    page.keyboard.press("Enter")
    try:
        page.wait_for_function(
            """() => {
              const roots = [document];
              document.querySelectorAll('gradio-app, .gradio-container').forEach(h => {
                if (h.shadowRoot) roots.push(h.shadowRoot);
              });
              for (const root of roots) {
                let el = root.getElementById('btn-confirm') || root.querySelector('[id="btn-confirm"]');
                if (!el) continue;
                const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
                if (btn && btn.disabled) return true;
              }
              return false;
            }""",
            timeout=LOCK_CHECK_TIMEOUT_MS,
        )
    except Exception:
        return "confirm never disabled during processing"
    locked = page.evaluate("""() => {
      function gradioBtn(id) {
        const roots = [document];
        document.querySelectorAll('gradio-app, .gradio-container').forEach(h => {
          if (h.shadowRoot) roots.push(h.shadowRoot);
        });
        for (const root of roots) {
          let el = root.getElementById(id) || root.querySelector('[id="' + id + '"]');
          if (!el) continue;
          return el.tagName === 'BUTTON' ? el : el.querySelector('button');
        }
        return null;
      }
      const btn = gradioBtn('btn-confirm');
      if (!btn || !btn.disabled) return false;
      const before = document.body.innerText;
      document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}));
      return document.body.innerText === before;
    }""")
    if not locked:
        return "Enter was not blocked while confirm disabled"
    try:
        page.wait_for_function(
            "() => document.body.innerText.includes('당신의 차례')",
            timeout=ROUND_TRIP_TIMEOUT_MS,
        )
    except Exception:
        return "enter locked test left game stuck in AI phase"
    return None


def _wait_exchange_done(page, min_exchange: int, timeout_ms: int = ROUND_TRIP_TIMEOUT_MS) -> None:
    page.wait_for_function(
        f"""() => {{
          const t = document.body.innerText;
          if ((t.includes('Round') && t.includes('결과')) || /\\d+\\s*\\/\\s*5000/.test(t)) return true;
          const m = t.match(/교환 (\\d+)\\/3/);
          return m && parseInt(m[1], 10) >= {min_exchange};
        }}""",
        timeout=timeout_ms,
    )


def test_exchange3_ai_before_round_end(page) -> str | None:
    """교환 1~2 빠르게, 교환 3에서 AI 응답 후 결과."""
    for n in range(2):
        page.keyboard.press("a")
        page.wait_for_timeout(150)
        page.keyboard.press("Enter")
        try:
            _wait_exchange_done(page, n + 2)
        except Exception:
            return f"exchange {n + 1}: round-trip timeout"
        page.wait_for_timeout(400)

    page.keyboard.press("s")
    page.wait_for_timeout(200)
    page.keyboard.press("Enter")
    info = page.evaluate("""() => {
      const t = document.body.innerText;
      return {
        aiPhase: t.includes('AI 응답 중'),
        userTurn: t.includes('당신의 차례'),
        roundResult: /Round\\s*\\d+\\s*결과/.test(t) || (t.includes('결과') && t.includes('TRY')),
        hasAudio: !!document.querySelector('#s3-exchange-audio audio, [id="s3-exchange-audio"] audio'),
        audioDuration: (() => {
          const a = document.querySelector('#s3-exchange-audio audio')
            || document.querySelector('[id="s3-exchange-audio"] audio');
          return a && isFinite(a.duration) ? a.duration : 0;
        })(),
      };
    }""")
    # 짧게 상태 전환만 관찰
    page.wait_for_timeout(300)
    info2 = page.evaluate("""() => {
      const t = document.body.innerText;
      return {
        aiPhase: t.includes('AI 응답 중'),
        roundResult: /Round\\s*\\d+\\s*결과/.test(t) || (t.includes('결과') && t.includes('TRY')),
        userTurn: t.includes('당신의 차례'),
      };
    }""")
    if info2["roundResult"] and not info["aiPhase"] and info2.get("userTurn") is False:
        return f"exchange 3: jumped to round end without AI phase: {info} -> {info2}"
    try:
        page.wait_for_function(
            "() => { const t=document.body.innerText;"
            "return (t.includes('Round') && t.includes('결과')) || /\\d+\\s*\\/\\s*5000/.test(t); }",
            timeout=ROUND_RESULT_TIMEOUT_MS,
        )
    except Exception:
        return f"exchange 3: result panel not shown, state={info2}"
    body3 = page.inner_text("body")
    if len(body3.strip()) < 60:
        return "exchange 3: white/empty screen after confirm"
    return None


def main() -> int:
    subprocess.run(
        ["sh", "-c", "lsof -ti:7860 | xargs kill -9 2>/dev/null || true"],
        check=False,
    )
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app.py")],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    errors: list[str] = []
    try:
        wait_server(proc)
        print("[test] server up")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            start_game(page)

            for name, fn in [
                ("keyboard j/k", test_keyboard_j_not_k),
                ("enter locked", test_enter_locked_during_ai),
            ]:
                # fresh notes for lock test needs new game - run lock after j/k uses backspace
                err = fn(page)
                if err:
                    errors.append(f"{name}: {err}")
                    print(f"[FAIL] {name}: {err}")
                else:
                    print(f"[PASS] {name}")

            # restart game for exchange 3 test
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            start_game(page)
            err = test_exchange3_ai_before_round_end(page)
            if err:
                errors.append(f"exchange3: {err}")
                print(f"[FAIL] exchange3: {err}")
            else:
                print("[PASS] exchange3 AI before round end")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    if errors:
        print("\n=== FAILED ===")
        for e in errors:
            print(" -", e)
        return 1
    print("\n=== ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
