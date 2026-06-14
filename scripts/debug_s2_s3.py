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
        page.wait_for_timeout(2500)
        page.get_by_role("button", name=re.compile("피아노")).click()
        page.wait_for_timeout(1500)
        page.get_by_role("button", name=re.compile("^▶ 시작")).click()
        page.wait_for_timeout(4000)
        info = page.evaluate("""() => ({
          text: document.body.innerText.slice(0, 600),
          hasUserTurn: document.body.innerText.includes('당신의 차례'),
          hasPiano: document.body.innerText.includes('옥타브'),
          screens: Array.from(document.querySelectorAll('.game-panel')).map(el => ({
            hide: el.classList.contains('hide'),
            h: el.offsetHeight,
          })),
        })""")
        print(info)
finally:
    proc.terminate()
