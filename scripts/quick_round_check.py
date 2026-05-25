#!/usr/bin/env python3
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"
SERVER_READY_TRIES = 80  # 40s
ROUND_TRIP_TIMEOUT_MS = 35000
RESULT_TIMEOUT_MS = 45000


def _exchange_num(page) -> int:
    v = page.evaluate(
        "() => parseInt((document.body.innerText.match(/교환 (\\d+)\\/3/) || [0, '0'])[1], 10)"
    )
    return int(v or 0)


def wait_after_confirm(page, min_exchange: int, timeout_ms: int = ROUND_TRIP_TIMEOUT_MS) -> None:
    page.wait_for_function(
        f"""() => {{
          const t = document.body.innerText;
          if ((t.includes('Round') && t.includes('결과')) || /\\d+\\s*\\/\\s*5000/.test(t)) return true;
          const m = t.match(/교환 (\\d+)\\/3/);
          return m && parseInt(m[1], 10) >= {min_exchange};
        }}""",
        timeout=timeout_ms,
    )


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

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            page = p.chromium.launch(headless=True).new_page()
            page.goto(URL, timeout=120000)
            page.wait_for_timeout(1200)
            page.get_by_role("button", name=re.compile("피아노")).click()
            page.wait_for_timeout(800)
            page.get_by_role("button", name=re.compile("시작")).click()
            page.wait_for_timeout(1200)
            for ex in range(1, 4):
                page.keyboard.press("a")
                page.wait_for_timeout(200)
                t0 = time.time()
                page.get_by_role("button", name=re.compile("확정")).click()
                if ex < 3:
                    wait_after_confirm(page, ex + 1)
                else:
                    page.wait_for_function(
                        "() => { const t=document.body.innerText;"
                        "return (t.includes('Round') && t.includes('결과')) || /\\d+\\s*\\/\\s*5000/.test(t); }",
                        timeout=RESULT_TIMEOUT_MS,
                    )
                body = page.inner_text("body")
                has_result = ("결과" in body and "Round" in body) or ("/5000" in body)
                print(
                    f"ex{ex}: {time.time()-t0:.1f}s "
                    f"ex={_exchange_num(page)} result={has_result}"
                )
                if has_result:
                    print("result OK")
                    return 0
            print("FAIL: no result after 3 exchanges")
            return 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
