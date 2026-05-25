"""Verify the parallax cosmos: two SVG layers, custom notes, distance-based activity."""
import json
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-orbit-outer", timeout=30000)
    page.wait_for_timeout(2000)  # let JS init + a couple frames

    info = page.evaluate(
        """() => {
          const outer = document.querySelector('.ra-orbit-outer');
          const inner = document.querySelector('.ra-orbit-inner');
          const oNotes = outer ? outer.querySelectorAll('.ra-orbit-note') : [];
          const iNotes = inner ? inner.querySelectorAll('.ra-orbit-note') : [];
          const syms = ['note-quarter','note-eighth','note-beamed','note-sixteenth'].map(id => !!document.getElementById(id));
          return {
            symbolsPresent: syms,
            outer: { count: oNotes.length, transform: outer ? outer.style.transform : null },
            inner: { count: iNotes.length, transform: inner ? inner.style.transform : null },
            sampleOuter: Array.from(oNotes).slice(0,2).map(n => ({ x: n.dataset.x, y: n.dataset.y, scale: n.dataset.scale, fill: n.getAttribute('fill'), transformAttr: n.getAttribute('transform') })),
            sampleInner: Array.from(iNotes).slice(0,2).map(n => ({ x: n.dataset.x, y: n.dataset.y, scale: n.dataset.scale, fill: n.getAttribute('fill'), transformAttr: n.getAttribute('transform') })),
          };
        }"""
    )
    print("init state:", json.dumps(info, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/cosmos_v2_idle.png", full_page=False)

    # Move mouse close to cosmos center
    box = page.locator(".ra-hero-cosmos").bounding_box()
    cx = box["x"] + box["width"]/2
    cy = box["y"] + box["height"]/2
    page.mouse.move(cx + 1, cy + 1)
    page.wait_for_timeout(2500)  # let activity ramp up + rotation accrue
    state_active = page.evaluate(
        """() => {
          const outer = document.querySelector('.ra-orbit-outer');
          const inner = document.querySelector('.ra-orbit-inner');
          return {
            outer: outer.style.transform,
            inner: inner.style.transform,
            firstOuterNote: outer.querySelector('.ra-orbit-note').getAttribute('transform'),
            firstInnerNote: inner.querySelector('.ra-orbit-note').getAttribute('transform'),
          };
        }"""
    )
    print("after hover ~2.5s:", json.dumps(state_active, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/cosmos_v2_hover.png", full_page=False)

    # Move mouse far away
    page.mouse.move(cx + 1200, cy)
    page.wait_for_timeout(2500)
    state_far = page.evaluate(
        """() => ({
          outer: document.querySelector('.ra-orbit-outer').style.transform,
          inner: document.querySelector('.ra-orbit-inner').style.transform,
        })"""
    )
    print("after mouse far:", json.dumps(state_far, indent=2, ensure_ascii=False))
    b.close()
