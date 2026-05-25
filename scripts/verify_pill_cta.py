"""Inspect Start session / Begin round buttons after change."""
import re, json
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-nav-logo", timeout=30000)
    page.wait_for_timeout(1500)

    def btn_info(label):
        return page.evaluate(
            """(lbl) => {
              const btns = Array.from(document.querySelectorAll('button'));
              const b = btns.find(x => (x.textContent||'').trim().startsWith(lbl));
              if (!b) return null;
              const cs = getComputedStyle(b);
              const before = getComputedStyle(b, '::before');
              const r = b.getBoundingClientRect();
              return {
                text: b.textContent.trim(),
                class: b.className,
                bg: cs.backgroundColor,
                color: cs.color,
                w: Math.round(r.width), h: Math.round(r.height),
                beforeContent: before.content,
                beforeColor: before.color,
                beforeFontSize: before.fontSize,
                beforeWidth: before.width,
              };
            }""",
            label,
        )

    s1_info = {
        "Start session": btn_info("Start session"),
        "Humming mode": btn_info("Humming mode"),
    }
    page.screenshot(path=f"{OUT}/s1_pill.png", full_page=False)
    # Click Start to reach S2
    page.get_by_role("button", name=re.compile("Start session")).click()
    page.wait_for_timeout(900)
    s2_info = { "Begin round": btn_info("Begin round") }
    page.screenshot(path=f"{OUT}/s2_pill.png", full_page=False)

    print("S1:", json.dumps(s1_info, indent=2, ensure_ascii=False))
    print("S2:", json.dumps(s2_info, indent=2, ensure_ascii=False))
    b.close()
