"""Play through the actual game to reach S5 and screenshot."""
import re, json, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
OUT = "/tmp/respondai_buttons"

def wait_for_your_turn(page, timeout=240000):
    """Wait until the 'YOUR TURN' badge appears (= AI generation finished)."""
    page.wait_for_function(
        "() => document.body.innerText.toUpperCase().includes('YOUR TURN')",
        timeout=timeout,
    )

def wait_until_finished_round(page, timeout=240000):
    """Wait until either 'See result' button or 'Next round' button appears AND is enabled."""
    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('button')).some(b => /(See result|Next round)/.test(b.textContent||'') && !b.disabled)",
        timeout=timeout,
    )

def click_by_text(page, label):
    """Click first enabled button whose text contains label."""
    page.evaluate(
        """(lbl) => {
          const btns = Array.from(document.querySelectorAll('button'));
          const b = btns.find(x => (x.textContent||'').includes(lbl) && !x.disabled);
          if (b) b.click();
          else throw new Error('button not found: ' + lbl);
        }""",
        label,
    )

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".ra-nav-logo", timeout=30000)
    page.wait_for_timeout(1500)

    click_by_text(page, "Start session")
    page.wait_for_timeout(1000)

    for round_idx in range(1, 3):
        print(f"--- Round {round_idx} ---")
        click_by_text(page, "Begin round")
        page.wait_for_timeout(1000)
        wait_for_your_turn(page)

        for ex in range(3):
            print(f"  exchange {ex+1}")
            page.keyboard.press("a"); page.wait_for_timeout(150)
            page.keyboard.press("d"); page.wait_for_timeout(150)
            page.keyboard.press("g"); page.wait_for_timeout(150)
            page.keyboard.press("Enter")
            if ex < 2:
                wait_for_your_turn(page)
            else:
                wait_until_finished_round(page)
                page.wait_for_timeout(700)

        print("  click See result")
        click_by_text(page, "See result")
        page.wait_for_timeout(1500)
        if round_idx < 2:
            print("  click Next round")
            click_by_text(page, "Next round")
            page.wait_for_timeout(1500)
        else:
            page.wait_for_timeout(3000)

    # Now on S5
    page.screenshot(path=f"{OUT}/s5_live_full.png", full_page=False)

    info = page.evaluate(
        """() => {
          const stage = document.querySelector('.s5-stage');
          const rounds = document.querySelector('.s5-rounds');
          const half = document.querySelector('.s5-halfsphere');
          const actions = document.querySelector('.s5-actions');
          const btns = Array.from(document.querySelectorAll('.s5-actions button, .s5-actions .game-pill'));
          const visible = (el) => { if (!el) return null; const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), top: Math.round(r.top), bottom: Math.round(r.bottom) }; };
          const styled = (el, props) => { if (!el) return null; const cs = getComputedStyle(el); return Object.fromEntries(props.map(p => [p, cs[p]])); };
          return {
            stage: visible(stage),
            rounds: { ...visible(rounds), text: (rounds||{}).textContent && rounds.textContent.trim(), fs: styled(rounds, ['fontSize','letterSpacing']) },
            halfsphere: visible(half),
            actions: visible(actions),
            buttons: btns.map(b => ({ ...visible(b), text: b.textContent.trim(), classes: b.className })),
          };
        }"""
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))
    page.screenshot(path=f"{OUT}/s5_live.png", clip={"x": 0, "y": 0, "width": 1400, "height": 900})
    b.close()
