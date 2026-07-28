"""Browser smoke test for the V7.19 gomoku deployment.

Run with:
  conda run -n pytorch python tests/browser_v719_smoke.py
while serve.py is listening on 127.0.0.1:8000.
"""

import os

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get(
    "SKYZERO_WEB_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
URL = f"{BASE_URL}/gomoku.html"


def wait_ready(page):
    page.wait_for_function(
        """() => {
          const el = document.getElementById("loading_overlay");
          return el && getComputedStyle(el).display === "none";
        }""",
        timeout=120_000,
    )


def main():
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: errors.append(
            f"console {msg.type}: {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        wait_ready(page)
        assert page.locator("#h_nn_futurepos_8").count() == 0
        assert page.locator("#h_nn_futurepos_32").count() == 0
        assert page.locator("#h_nn_optimistic_policy").is_visible()
        assert page.locator("#h_mcts_play_selection").is_visible()
        assert not page.locator("#h_mcts_visits").is_visible()
        page.locator(
            '.pin-btn[data-target="h_mcts_play_selection"]').click()
        assert page.locator(
            '.pin-btn[data-target="h_mcts_play_selection"]').get_attribute(
                "aria-pressed") == "true"
        page.locator("#heat_more_btn").click()
        assert page.locator("#h_mcts_visits").is_visible()

        page.locator("#model_trigger").click()
        options = page.locator("#model_menu .cs-option")
        assert options.count() == 6
        assert [options.nth(i).get_attribute("data-id") for i in range(6)] == [
            "lv1", "lv2", "lv3", "lv4", "lv5", "lv6"]

        # Exercise every ONNX file, including session construction and output
        # contracts. The current/default model is harmlessly reloaded too.
        for model_id in ("lv1", "lv2", "lv3", "lv4", "lv5", "lv6"):
            if not page.locator("#model_menu").is_visible():
                page.locator("#model_trigger").click()
            page.locator(f'#model_menu .cs-option[data-id="{model_id}"]').click()
            wait_ready(page)
            assert page.locator("#model_trigger_label").inner_text().startswith(
                model_id.upper())

        # V7.19 runtime board range and all three rule buttons.
        slider = page.locator("#size_input")
        assert slider.get_attribute("min") == "11"
        assert slider.get_attribute("max") == "15"
        for size in (11, 13, 15):
            slider.evaluate(
                """(el, value) => {
                  el.value = String(value);
                  el.dispatchEvent(new Event("input", {bubbles:true}));
                  el.dispatchEvent(new Event("change", {bubbles:true}));
                }""",
                size,
            )
            page.wait_for_timeout(100)
            assert page.locator("#size_value").inner_text() == str(size)

        for rule in ("standard", "renju", "freestyle"):
            if not page.locator("#settings_pop").is_visible():
                page.locator("#settings_btn").click()
            page.locator(f'.seg-btn[data-rule="{rule}"]').click()
            assert page.locator(
                f'.seg-btn[data-rule="{rule}"]').get_attribute(
                    "aria-pressed") == "true"

        # Position-edit path: enter, place a stone, finish, and return to search.
        if not page.locator("#settings_pop").is_visible():
            page.locator("#settings_btn").click()
        page.locator("#edit_btn").click()
        assert page.locator("#edit_toolbar").is_visible()
        box = page.locator("#board").bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2,
                         box["y"] + box["height"] / 2)
        page.locator("#edit_done_btn").click()
        assert not page.locator("#edit_toolbar").is_visible()

        # Compact layout must retain usable board and controls.
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(250)
        assert page.locator("#board").bounding_box()["width"] > 250
        assert page.locator("#settings_btn").is_visible()

        tree_page = browser.new_page(viewport={"width": 1280, "height": 800})
        tree_page.on("console", lambda msg: errors.append(
            f"tree console {msg.type}: {msg.text}")
            if msg.type == "error" else None)
        tree_page.on("pageerror",
                     lambda exc: errors.append(f"tree pageerror: {exc}"))
        tree_page.goto(f"{BASE_URL}/mcts-tree.html",
                       wait_until="domcontentloaded", timeout=120_000)
        tree_page.wait_for_function(
            """() => {
              const s = document.getElementById("status");
              return s && s.textContent.includes("根总访问");
            }""",
            timeout=120_000,
        )
        assert tree_page.locator("#size option").count() == 5
        assert tree_page.locator("#model option").count() == 6
        tree_page.locator("#btnStep").click()
        tree_page.wait_for_function(
            """() => document.getElementById("status")
              .textContent.startsWith("1 次模拟")""",
            timeout=120_000,
        )

        browser.close()

    if errors:
        raise AssertionError("\n".join(errors))
    print("browser-v719-smoke: ok")


if __name__ == "__main__":
    main()
