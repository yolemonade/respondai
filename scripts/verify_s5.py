"""Render S5 standalone for quick visual verification."""
import sys, os, tempfile, pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as gradio_app  # noqa
from playwright.sync_api import sync_playwright

def _mock_ex(total, kc=0.75, rs=0.7, mu=0.6, cb=0.65):
    return {
        "total": total,
        "key_consistency": kc, "rhythm_similarity": rs,
        "motif_usage": mu, "creativity_bonus": cb,
        "feedback": "",
        "raw": {"key_consistency": kc, "rhythm_similarity": rs, "motif_usage": mu, "creativity_bonus": cb},
    }

mock_state = {
    "round_results": [
        {"round_num": 1, "key": "C major",
         "exchange_scores": [_mock_ex(700), _mock_ex(720), _mock_ex(740)],
         "r5_motif_bonus": False},
        {"round_num": 2, "key": "G major",
         "exchange_scores": [_mock_ex(820, 0.85, 0.8), _mock_ex(800, 0.85, 0.8), _mock_ex(780, 0.85, 0.8)],
         "r5_motif_bonus": True},
    ]
}

html_body = gradio_app.s5_html(mock_state)
APP_CSS = gradio_app.APP_CSS

# Wrap in a standalone HTML doc that mimics the panel-s5 + s5-body + s5-actions layout
button_label = "\u21bb  Play again"
standalone = f"""<!doctype html>
<html><head>
  <meta charset="utf-8" />
  <style>{APP_CSS}</style>
  <style>
    /* Force dark-panel context */
    html, body, gradio-app {{ background: #0E0E0E !important; }}
    body {{ margin: 0; padding: 0; min-height: 100vh; display: flex; align-items: stretch; justify-content: center; }}
    .gradio-container {{ width: 100%; max-width: 1000px; }}
    .panel-s5 {{
      background: #0E0E0E; color: #F0F0F0;
      min-height: 100vh; display: flex; flex-direction: column;
      position: relative; overflow: hidden;
    }}
    html, body {{ height: 100%; }}
    .panel-s5 {{ min-height: 1080px !important; height: 1080px !important; }}
    .s5-body {{
      flex: 1 1 auto;
      overflow: hidden !important;
      position: relative;
      display: flex; flex-direction: column;
      min-height: 0;
    }}
    .s5-body > .s5-stage {{
      flex: 1 1 auto;
      height: 100% !important;
    }}
    .s5-actions {{ display: flex !important; align-items: center; }}
    .s5-actions button {{ all: revert; cursor: pointer; font-family: 'Inter', -apple-system, sans-serif; }}
  </style>
</head>
<body>
  <div class="gradio-container">
    <div class="panel-s5 game-panel">
      <div class="screen-body s5-body">{html_body}</div>
      <div class="screen-actions s5-actions">
        <button class="game-pill game-pill-primary" type="button">{button_label}</button>
      </div>
    </div>
  </div>
</body></html>
"""

p = pathlib.Path("/tmp/respondai_buttons/s5_preview.html")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(standalone, encoding="utf-8")

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    ctx = b.new_context(viewport={"width": 1400, "height": 900})
    page = ctx.new_page()
    page.set_viewport_size({"width": 1400, "height": 1080})
    page.goto(f"file://{p}")
    page.wait_for_timeout(800)
    page.screenshot(path="/tmp/respondai_buttons/s5_preview.png", full_page=False)
    # Measurements
    info = page.evaluate(
        """() => {
          const btn = document.querySelector('.s5-actions .game-pill');
          const rounds = document.querySelector('.s5-rounds');
          const half = document.querySelector('.s5-halfsphere');
          const actions = document.querySelector('.s5-actions');
          const r = el => { const b = el.getBoundingClientRect(); return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height), top: Math.round(b.top), bottom: Math.round(b.bottom) }; };
          return {
            button: { ...r(btn), text: btn.textContent.trim() },
            rounds: { ...r(rounds), text: rounds.textContent.trim(), fontSize: getComputedStyle(rounds).fontSize },
            halfsphere: r(half),
            actions: r(actions),
            overlap: r(half).bottom > r(btn).top ? r(half).bottom - r(btn).top : 0,
          };
        }"""
    )
    import json
    print(json.dumps(info, indent=2, ensure_ascii=False))
    b.close()
