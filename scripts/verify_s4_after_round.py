"""Play through R1 (3 exchanges) and assert S4 round result is visible (no white screen)."""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:7860"


def up() -> bool:
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except Exception:
        return False


def main() -> int:
    proc = None
    if not up():
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env)
        for _ in range(120):
            if up():
                break
            if proc.poll():
                return 2
            time.sleep(0.5)

    try:
        with sync_playwright() as p:
            page = p.chromium.launch(headless=True).new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_selector("#btn-piano-start", timeout=30_000)
            page.click("#btn-piano-start")
            page.wait_for_timeout(800)
            page.get_by_role("button", name=re.compile("Begin round", re.I)).click()
            page.wait_for_timeout(1000)

            for ex in range(3):
                page.keyboard.press("a")
                page.wait_for_timeout(120)
                page.keyboard.press("a")
                page.wait_for_timeout(120)
                page.click("#btn-confirm")
                if ex < 2:
                    page.wait_for_function(
                        "() => document.querySelector('.panel-s3:not(.hide)') && "
                        "document.body.innerText.includes('YOUR TURN')",
                        timeout=180_000,
                    )
                else:
                    page.wait_for_selector(".ra-result-card", timeout=180_000)
                    page.wait_for_timeout(800)
                    page.evaluate("() => window.raShowScreen && window.raShowScreen('S4')")
                    page.wait_for_timeout(400)

            info = page.evaluate(
                """() => {
                  const panels = Array.from(document.querySelectorAll('.game-panel')).map(el => ({
                    cls: el.className,
                    h: el.offsetHeight,
                    hidden: el.classList.contains('hide'),
                  }));
                  const root = document.getElementById('ra-screen-nav');
                  const inp = root && root.querySelector('textarea, input[type="text"]');
                  return {
                    panels,
                    navRoot: !!root,
                    navValue: inp ? inp.value : null,
                    raShowScreen: typeof window.raShowScreen,
                    hasResult: !!document.querySelector('.ra-result-card'),
                    hasNextRound: Array.from(document.querySelectorAll('button')).some(b =>
                      /Next round/i.test(b.innerText)),
                    snippet: document.body.innerText.slice(0, 500),
                  };
                }"""
            )
            print(info)
            s4_visible = any(
                "panel-s4" in p["cls"] and not p["hidden"] and p["h"] > 50
                for p in info["panels"]
            )
            ok = info["hasResult"] and info["hasNextRound"] and s4_visible
            print("s4_visible:", s4_visible)
            if not ok:
                page.screenshot(path=str(ROOT / "scripts" / "_s4_fail.png"))
                print("FAIL — screenshot scripts/_s4_fail.png")
                return 1
            print("OK — S4 round result visible")
            return 0
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
