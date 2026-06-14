"""Verify piano key labels — black keys must now have both note name + key shortcut."""
import re, json
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 1000})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-nav-logo", timeout=30000)
    page.wait_for_timeout(1500)

    # Navigate S1 → S2 → S3 (Begin round)
    page.evaluate(
        """() => {
          const btns = Array.from(document.querySelectorAll('button'));
          const b = btns.find(x => (x.textContent||'').includes('Start session') && !x.disabled);
          if (b) b.click();
        }"""
    )
    page.wait_for_timeout(900)
    page.evaluate(
        """() => {
          const btns = Array.from(document.querySelectorAll('button'));
          const b = btns.find(x => (x.textContent||'').includes('Begin round') && !x.disabled);
          if (b) b.click();
        }"""
    )
    page.wait_for_timeout(1200)
    page.wait_for_selector(".ra-key", timeout=20000)
    page.wait_for_timeout(500)

    info = page.evaluate(
        """() => {
          const keys = Array.from(document.querySelectorAll('.ra-key'));
          return keys.map(k => ({
            midi: k.dataset.midi,
            color: k.classList.contains('ra-key-black') ? 'black' : 'white',
            label: (k.querySelector('.ra-key-label') || {}).textContent || '',
            shortcut: (k.querySelector('.ra-key-shortcut') || {}).textContent || '',
          }));
        }"""
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/piano_keys.png", clip={"x": 0, "y": 250, "width": 1400, "height": 480})
    b.close()
