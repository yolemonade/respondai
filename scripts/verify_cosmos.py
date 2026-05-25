"""Inspect S1 hero cosmos: orb + svg orbit + hover-spin behavior."""
import json
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-hero-cosmos", timeout=30000)
    page.wait_for_timeout(1500)

    cosmos = page.evaluate(
        """() => {
          const c = document.querySelector('.ra-hero-cosmos');
          const o = document.querySelector('.ra-hero-orbit');
          const orb = document.querySelector('.ra-hero-cosmos .ra-hero-orb');
          const notes = Array.from(document.querySelectorAll('.ra-hero-orbit .ra-orbit-note'));
          const cs = getComputedStyle(o);
          return {
            cosmos: c ? { w: c.offsetWidth, h: c.offsetHeight } : null,
            orb:    orb ? { w: orb.offsetWidth, h: orb.offsetHeight } : null,
            orbitState: cs.animationPlayState,
            orbitAnimation: cs.animationName,
            orbitDuration: cs.animationDuration,
            noteCount: notes.length,
            noteSample: notes.slice(0,3).map(n => ({ text: n.textContent, fill: n.getAttribute('fill'), x: n.getAttribute('x'), y: n.getAttribute('y') })),
          };
        }"""
    )
    print("cosmos state (idle):", json.dumps(cosmos, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/s1_cosmos_idle.png", full_page=False)

    # Hover to trigger orbit
    cos_box = page.locator(".ra-hero-cosmos").bounding_box()
    page.mouse.move(cos_box["x"] + cos_box["width"]/2, cos_box["y"] + cos_box["height"]/2)
    page.wait_for_timeout(500)
    hover_state = page.evaluate("() => getComputedStyle(document.querySelector('.ra-hero-orbit')).animationPlayState")
    print("orbit play state on hover:", hover_state)
    page.wait_for_timeout(3500)  # let it spin a bit
    page.screenshot(path=f"{OUT}/s1_cosmos_hover.png", full_page=False)

    b.close()
