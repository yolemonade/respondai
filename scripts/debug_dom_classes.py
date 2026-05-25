#!/usr/bin/env python3
import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["sh", "-c", "lsof -ti:7860 | xargs kill -9 2>/dev/null || true"], check=False)
proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=str(ROOT))
time.sleep(22)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page()
        page.goto("http://127.0.0.1:7860", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(3000)

        def snap(label):
            info = page.evaluate("""() => {
              const cls = (s) => Array.from(document.querySelectorAll(`[class*="${s}"]`))
                .slice(0, 8).map(el => ({
                  tag: el.tagName,
                  cls: el.className.toString().slice(0, 120),
                  hide: el.classList.contains('hide'),
                  h: el.offsetHeight,
                  text: el.innerText?.slice(0, 80),
                }));
              return {
                body: document.body.innerText.slice(0, 400),
                gamePanel: cls('game-panel'),
                gameStage: cls('game-stage'),
                gameScreen: cls('game-screen'),
                hideEls: document.querySelectorAll('.hide').length,
                buttons: Array.from(document.querySelectorAll('button')).map(b => b.innerText.trim()).slice(0, 15),
              };
            }""")
            print(f"\n=== {label} ===")
            print(info)

        snap("S1")
        page.get_by_role("button", name=re.compile("피아노")).click()
        page.wait_for_timeout(2500)
        snap("S2")
        page.get_by_role("button", name=re.compile("^▶ 시작")).click()
        page.wait_for_timeout(4000)
        snap("S3?")
finally:
    proc.terminate()
