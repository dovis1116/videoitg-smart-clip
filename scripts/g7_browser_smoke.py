"""Headless browser smoke for the documented local frontend/API split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    started = time.perf_counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        # The fixture is intentionally not a playable video.  Prevent the
        # headless media pipeline from blocking the renderer while still
        # exercising upload, polling, candidate selection and feedback.
        page.add_init_script("Object.defineProperty(HTMLMediaElement.prototype, 'src', {set() {}, get() { return ''; }});")
        page.goto(args.frontend_url, wait_until="networkidle")
        assert page.locator("h1").inner_text() == "长视频语义片段检索"
        page.locator("#video-file").set_input_files(str(args.video.resolve()))
        page.locator("#query").fill("find the matching action")
        # Dispatch through the DOM so Playwright does not wait on the async
        # click handler's network work (the handler itself is intentionally
        # not awaited by the browser event loop).
        page.evaluate("document.querySelector('#submit').click()")
        page.locator(".prediction").first.wait_for(state="visible", timeout=15_000)
        page.locator(".prediction button").first.evaluate("el => el.click()")
        page.locator("#adjusted-start").fill("0.50")
        page.locator("#adjusted-end").fill("4.50")
        page.get_by_role("button", name="采用").evaluate("el => el.click()")
        page.locator("#feedback-message").filter(has_text="反馈已保存").wait_for(timeout=10_000)
        result = {
            "frontend_url": args.frontend_url,
            "title": page.title(),
            "task_meta": page.locator("#task-meta").inner_text(),
            "prediction_count": page.locator(".prediction").count(),
            "feedback_message": page.locator("#feedback-message").inner_text(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "human_user_test": False,
            "scope": "automated_headless_browser_smoke",
        }
        browser.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
