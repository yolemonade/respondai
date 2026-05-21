"""Detect which element creates the S1 scrollbar.

Spawns app.py (if not already running) and inspects scrollable elements + sizes
of key S1 containers. Prints a concise report.
"""
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


def is_up() -> bool:
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except Exception:
        return False


def main() -> int:
    proc: subprocess.Popen | None = None
    if not is_up():
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for _ in range(120):
            if is_up():
                break
            if proc.poll() is not None:
                print("[boot] app.py exited unexpectedly")
                return 2
            time.sleep(0.5)
    viewports = [
        (1440, 900),
        (1280, 800),
        (1366, 700),
        (1200, 720),
        (1100, 700),
        (1024, 640),
        (900, 580),
        (800, 600),
        (768, 600),
        (640, 800),
        (480, 800),
    ]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": viewports[0][0], "height": viewports[0][1]})
            page = ctx.new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector(".panel-s1", timeout=30_000)
            page.wait_for_timeout(900)

            for w, h in viewports:
                page.set_viewport_size({"width": w, "height": h})
                page.wait_for_timeout(250)
                summary = page.evaluate(
                    """
                    () => {
                      const all = document.querySelectorAll('*');
                      const scr = [];
                      all.forEach(el => {
                        const cs = getComputedStyle(el);
                        const oy = cs.overflowY, ox = cs.overflowX;
                        const vS = (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 1;
                        const hS = (ox === 'auto' || ox === 'scroll') && el.scrollWidth > el.clientWidth + 1;
                        if (vS || hS) scr.push({tag: el.tagName.toLowerCase(), cls: (el.className||'').toString().slice(0,110), id: el.id||'', sh: el.scrollHeight, ch: el.clientHeight, overflowY: oy});
                      });
                      return {docH: document.documentElement.scrollHeight, scr};
                    }
                    """
                )
                print(f"\n== viewport {w}x{h} | docH={summary['docH']} ==")
                if not summary["scr"]:
                    print("  (no scrollables)")
                for s in summary["scr"][:6]:
                    print(" ", s)

            page.set_viewport_size({"width": 1280, "height": 800})
            page.wait_for_timeout(200)
            data = page.evaluate(
                """
                () => {
                  const out = { docScroll: {sh: document.documentElement.scrollHeight, ch: document.documentElement.clientHeight, sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth},
                                bodyScroll: {sh: document.body.scrollHeight, ch: document.body.clientHeight},
                                scrollables: [],
                                key: {} };
                  // gather every scrollable element
                  const all = document.querySelectorAll('*');
                  all.forEach(el => {
                    const r = el.getBoundingClientRect();
                    const cs = getComputedStyle(el);
                    const oy = cs.overflowY, ox = cs.overflowX;
                    const vScroll = (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 1;
                    const hScroll = (ox === 'auto' || ox === 'scroll') && el.scrollWidth > el.clientWidth + 1;
                    if (vScroll || hScroll) {
                      out.scrollables.push({
                        tag: el.tagName.toLowerCase(),
                        cls: el.className && el.className.toString().slice(0, 120),
                        id: el.id || '',
                        sh: el.scrollHeight, ch: el.clientHeight, sw: el.scrollWidth, cw: el.clientWidth,
                        ox, oy,
                        rect: {x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height)},
                      });
                    }
                  });
                  // key containers
                  const wanted = ['.panel-s1', '.panel-s1 .screen-body', '.ra-title-wrap', '.ra-hero-cosmos', '.ra-orbit-outer', '.gradio-container', '.game-stage', '.s1-actions'];
                  for (const sel of wanted) {
                    const el = document.querySelector(sel);
                    if (!el) { out.key[sel] = null; continue; }
                    const r = el.getBoundingClientRect();
                    const cs = getComputedStyle(el);
                    out.key[sel] = {
                      rect: {x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height)},
                      sh: el.scrollHeight, ch: el.clientHeight,
                      overflowY: cs.overflowY, overflowX: cs.overflowX,
                    };
                  }
                  return out;
                }
                """
            )

            print("=== document ===")
            print(data["docScroll"])
            print("body:", data["bodyScroll"])

            print("\n=== scrollable elements (top-3) ===")
            for s in data["scrollables"][:6]:
                print(s)

            print("\n=== key containers ===")
            for k, v in data["key"].items():
                print(f"{k}: {v}")

            page.screenshot(path=str(ROOT / "scripts" / "_s1_scroll_debug.png"), full_page=False)
            print("\nscreenshot saved → scripts/_s1_scroll_debug.png")
            browser.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
