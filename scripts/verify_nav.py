"""Inspect S1 nav: bounding boxes of logo & links, check English labels and alignment."""
import json
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

    info = page.evaluate(
        """() => {
          const r = el => { const b = el.getBoundingClientRect(); const cs = getComputedStyle(el); return { text: el.textContent.trim(), x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height), center: Math.round(b.y + b.height/2), fs: cs.fontSize, lh: cs.lineHeight }; };
          const logo = document.querySelector('.ra-nav-logo');
          const links = Array.from(document.querySelectorAll('.ra-nav-link'));
          const nav = document.querySelector('.ra-nav');
          const navBox = nav.getBoundingClientRect();
          const hero = document.querySelector('.ra-title-wrap');
          const heroBox = hero ? hero.getBoundingClientRect() : null;
          return {
            nav: { x: Math.round(navBox.x), y: Math.round(navBox.y), w: Math.round(navBox.width), h: Math.round(navBox.height) },
            hero: heroBox ? { x: Math.round(heroBox.x), w: Math.round(heroBox.width) } : null,
            logo: r(logo),
            links: links.map(r),
          };
        }"""
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/s1_nav.png", clip={"x": 0, "y": 0, "width": 1400, "height": 220})
    page.screenshot(path=f"{OUT}/s1_full.png", full_page=False)
    print("screenshots:", OUT)
    b.close()
