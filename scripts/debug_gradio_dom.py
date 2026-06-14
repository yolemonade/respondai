#!/usr/bin/env python3
import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860"

proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=str(ROOT))
time.sleep(25)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(4000)
        on_s1 = page.evaluate("""() => ({
          nk60: !!document.getElementById('nk-60'),
          layerBtns: document.querySelectorAll('.note-input-layer').length,
          cButtons: Array.from(document.querySelectorAll('button')).filter(b => b.textContent.trim()==='C').length,
        })""")
        print("on S1:", on_s1)
        page.get_by_role("button", name=re.compile("피아노")).click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name=re.compile("시작")).click()
        page.wait_for_timeout(2000)
        info = page.evaluate("""() => ({
          bridge: document.getElementById('note-bridge'),
          bridgeBtn: document.querySelector('#note-bridge button'),
          hiddenBlocks: document.querySelectorAll('.hidden-bridge').length,
          hiddenBtns: Array.from(document.querySelectorAll('.hidden-bridge button')).map(b => b.innerText),
          allIds: Array.from(document.querySelectorAll('[id]')).map(e => e.id).slice(0, 40),
          buttons: Array.from(document.querySelectorAll('button')).map(b => ({
            t: b.innerText.trim().slice(0, 20),
            id: b.id,
            pid: b.parentElement?.id,
            cls: b.parentElement?.className?.slice?.(0, 60),
          })),
          noteCmd: !!document.getElementById('note-cmd'),
          textareas: Array.from(document.querySelectorAll('textarea')).map(t => ({
            id: t.id, pid: t.parentElement?.id, hidden: t.offsetParent === null
          })),
          continueBtn: Array.from(document.querySelectorAll('button')).filter(b => b.innerText.trim() === 'continue').length,
          bodyHas16: document.body.innerText.match(/[1-9]\\/16/)?.[0],
          respondAI: typeof window.respondAI,
        })""")
        print("DOM:", info)
        page.keyboard.press("a")
        page.wait_for_timeout(1000)
        page.evaluate("() => { window._pendingNote = 60; window.respondAI && window.respondAI.sendNote(60); }")
        page.wait_for_timeout(2000)
        page.keyboard.press("a")
        page.wait_for_timeout(800)
        page.keyboard.press("a")
        page.wait_for_timeout(1500)
        after = page.evaluate("""() => ({
          hasNotes: /[1-9]\\/16/.test(document.body.innerText),
          noteLine: (document.body.innerText.match(/[^\\n]*\\/16[^\\n]*/)||[''])[0].slice(0,80),
        })""")
        print("after key a:", after)
finally:
    proc.terminate()
