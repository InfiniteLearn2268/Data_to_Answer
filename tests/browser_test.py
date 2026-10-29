"""Drives the real page in headless Chromium: upload a big file with the file picker, watch progress,
then ask a normal question and an ambiguous one (clarification flow)."""
import sys, time
from playwright.sync_api import sync_playwright
path, shots = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1000, "height": 800})
    errors = []; pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("http://127.0.0.1:8000/")
    pg.screenshot(path=f"{shots}/1_pick.png")
    t = time.time(); pg.set_input_files("#file", path)
    pg.wait_for_selector("#s-load:not(.hidden)")
    seen = set()
    while pg.is_visible("#s-load"):
        st = pg.inner_text("#loadStage"); seen.add(st)
        if "Uploading" in st and "up" not in seen: pg.screenshot(path=f"{shots}/2_uploading.png"); seen.add("up")
        time.sleep(0.15)
    print("stages seen on page:", sorted(s for s in seen if s != "up"), f"| total {time.time()-t:.0f}s")
    pg.wait_for_selector("#s-chat:not(.hidden)")
    print("description shown:", pg.inner_text("#chatDesc")[:90])
    def ask(q, wait_text):
        pg.fill("#q", q); pg.click("#send"); pg.wait_for_function(f"document.querySelector('#log').innerText.includes({wait_text!r})", timeout=60000)
    ask("total amount by region", "RESULT"); print("answer 1 rendered")
    ask("an ambiguous question", "Which region"); print("clarification bubble shown, placeholder =", pg.get_attribute("#q", "placeholder"))
    ask("Central", "RESULT"); print("answer after clarification rendered")
    pg.screenshot(path=f"{shots}/3_chat.png")
    pg.reload(); pg.wait_for_selector("#s-chat:not(.hidden)", timeout=10000); print("session restored after page refresh")
    print("JS errors:", errors or "none")
    b.close()
