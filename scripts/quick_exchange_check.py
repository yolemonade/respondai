#!/usr/bin/env python3
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"
SERVER_READY_TRIES = 80  # 80 * 0.5s = 40s
EXCHANGE_TIMEOUT_MS = 35000
RETURN_TIMEOUT_MS = 15000


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
    try:
        for _ in range(SERVER_READY_TRIES):
            if proc.poll():
                print(proc.stdout.read()[:2000])
                return 1
            try:
                urllib.request.urlopen(URL, timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("server timeout")
            return 1

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            page = p.chromium.launch(headless=True).new_page()
            page.goto(URL, timeout=120000)
            page.wait_for_timeout(1200)
            page.get_by_role("button", name=re.compile("피아노")).click()
            page.wait_for_timeout(800)
            page.get_by_role("button", name=re.compile("시작")).click()
            page.wait_for_timeout(1200)
            page.keyboard.press("a")
            page.wait_for_timeout(300)
            t0 = time.time()
            page.get_by_role("button", name=re.compile("확정")).click()
            page.wait_for_function(
                """() => {
                  const t = document.body.innerText;
                  return t.includes('당신의 차례') || t.includes('AI 응답');
                }""",
                timeout=EXCHANGE_TIMEOUT_MS,
            )
            t_ai = time.time() - t0
            page.wait_for_function(
                "() => document.body.innerText.includes('당신의 차례')",
                timeout=RETURN_TIMEOUT_MS,
            )
            t_back = time.time() - t0
            ex = page.evaluate(
                "() => (document.body.innerText.match(/교환 (\\d+)\\/3/)||[])[1]"
            )
            print(f"AI phase: {t_ai:.1f}s")
            print(f"User turn back: {t_back:.1f}s (exchange {ex}/3)")
            ok = ex == "2"
            print("PASS" if ok else "FAIL")
            return 0 if ok else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
