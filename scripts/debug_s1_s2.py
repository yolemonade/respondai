#!/usr/bin/env python3
import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=str(ROOT))
time.sleep(22)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page()
        page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3000)
        before = page.evaluate("""() => ({
          text: document.body.innerText.slice(0, 500),
          gameScreens: document.querySelectorAll('.game-screen').length,
          visibleScreens: Array.from(document.querySelectorAll('.game-screen')).filter(el => el.offsetHeight > 0 && el.offsetParent !== null).length,
        })""")
        print("before:", before)
        page.get_by_role("button", name=re.compile("피아노")).click()
        page.wait_for_timeout(3000)
        after = page.evaluate("""() => ({
          text: document.body.innerText.slice(0, 800),
          gameScreens: document.querySelectorAll('.game-screen').length,
          visibleScreens: Array.from(document.querySelectorAll('.game-screen')).filter(el => el.offsetHeight > 0 && el.offsetParent !== null).length,
          hasRound: document.body.innerText.includes('Round'),
          hasStart: document.body.innerText.includes('시작'),
          hideCount: document.querySelectorAll('.hide').length,
        })""")
        print("after:", after)
finally:
    proc.terminate()
