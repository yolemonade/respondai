"""Quick visual + DOM check of redesigned buttons and S1 nav."""
import re, time, json
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"
import os; os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-nav-logo", timeout=30000)
    page.wait_for_timeout(1500)

    # S1 — nav check
    nav_html = page.eval_on_selector(".ra-nav", "el => el.outerHTML")
    logo_size = page.eval_on_selector(".ra-nav-logo", "el => getComputedStyle(el).fontSize")
    sign_in_exists = page.locator(".ra-nav-cta").count()
    print("S1 nav logo fontSize:", logo_size)
    print("S1 SIGN IN count (expect 0):", sign_in_exists)
    print("S1 nav HTML snippet:", nav_html[:400])

    page.screenshot(path=f"{OUT}/s1.png", full_page=False)

    # S2
    page.get_by_role("button", name=re.compile("Start session")).click()
    page.wait_for_timeout(900)
    page.screenshot(path=f"{OUT}/s2.png", full_page=False)

    # S3
    page.get_by_role("button", name=re.compile("Begin round")).click()
    page.wait_for_timeout(900)
    # Check S3 buttons style
    btn_info = page.evaluate(
        """() => {
          const out = {};
          const sel = ['Undo','Preview','Confirm','See result','Play again'];
          const btns = Array.from(document.querySelectorAll('button'));
          for (const t of sel) {
            const b = btns.find(x => x.textContent && x.textContent.trim().startsWith(t.charAt(0)) || (x.textContent||'').includes(t));
            if (b) {
              const cs = getComputedStyle(b);
              out[t] = { text: b.textContent.trim(), bg: cs.backgroundColor, color: cs.color, radius: cs.borderRadius, class: b.className };
            }
          }
          return out;
        }"""
    )
    print("S3 button styles:")
    print(json.dumps(btn_info, indent=2))
    page.screenshot(path=f"{OUT}/s3.png", full_page=False)

    # Play one key + Enter to start an exchange (so we can later view S4/S5 inline cards)
    page.keyboard.press("a")
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    page.screenshot(path=f"{OUT}/s3_after_enter.png", full_page=False)

    print("done — screenshots in", OUT)
    browser.close()
