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
        page.locator(".difficulty-option").first.wait_for(
            state="visible", timeout=120_000)
        page.locator(".difficulty-option").first.click()
        wait_ready(page)
        assert page.locator("#think_trigger_val").inner_text() == "0.5秒"
        assert page.locator("#think_time_range").input_value() == "0"
        page.locator("#think_trigger").click()
        assert page.locator("#think_pop").is_visible()
        trigger_box = page.locator("#think_trigger").bounding_box()
        pop_box = page.locator("#think_pop").bounding_box()
        assert trigger_box and pop_box
        assert abs(
            trigger_box["x"] + trigger_box["width"] / 2
            - pop_box["x"] - pop_box["width"] / 2
        ) < 1
        assert page.locator("#search_toggle").is_checked()
        assert page.locator("#thinking_toggle_label").inner_text() == "启用思考"
        page.locator("#search_toggle").evaluate("(input) => input.click()")
        assert not page.locator("#search_toggle").is_checked()
        assert page.locator("#thinking_toggle_label").inner_text() == "关闭"
        assert page.locator("#think_trigger_val").inner_text() == "关闭"
        assert page.locator("#think_time_range").is_disabled()
        page.wait_for_function(
            """() => state && state.policy_only === true
              && state.policy_prior
              && Math.max(...state.policy_prior.flat()) > 0""",
            timeout=120_000,
        )
        assert page.locator("#cand_legend").inner_text() == "策略先验"
        assert page.locator("#cand_list .cand-row").count() > 0
        assert page.evaluate(
            """() => {
              const row = document.querySelector("#cand_list .cand-row");
              return row.querySelector(".cand-coord").textContent
                === coordLabel(Number(row.dataset.r), Number(row.dataset.c));
            }"""
        )
        assert page.evaluate(
            """() => {
              const row = document.querySelector("#cand_list .cand-row");
              const shown = state.policy_prior[
                Number(row.dataset.r)][Number(row.dataset.c)];
              return Math.abs(
                shown - Math.max(...state.policy_prior.flat())) < 1e-7;
            }"""
        )
        assert page.evaluate(
            """() => {
              const main = state.nn_policy.flat();
              const optimistic = state.nn_optimistic_policy.flat();
              const prior = state.policy_prior.flat();
              const blended = main.map((value, i) =>
                value > 0 && optimistic[i] > 0
                  ? Math.pow(value, 0.8)
                    * Math.pow(optimistic[i], 0.2)
                  : 0);
              const total = blended.reduce((sum, value) => sum + value, 0);
              return Math.max(...prior.map((value, i) =>
                Math.abs(value - blended[i] / total))) < 1e-6;
            }"""
        )
        page.locator("#search_toggle").evaluate("(input) => input.click()")
        assert page.locator("#think_trigger_val").inner_text() == "0.5秒"
        assert not page.locator("#think_time_range").is_disabled()
        pda_range = page.locator("#pda_range")
        assert pda_range.get_attribute("min") == "-1"
        assert pda_range.get_attribute("max") == "1"
        assert pda_range.get_attribute("step") == "0.1"
        assert pda_range.input_value() == "0.5"
        assert page.locator("#pda_value").count() == 0
        pda_range.fill("0.1")
        assert pda_range.input_value() == "0"
        assert page.evaluate("localStorage.getItem('skz_pda')") == "0"
        page.wait_for_function("() => state && state.pda === 0",
                               timeout=120_000)
        page.locator("#think_trigger").click()
        assert page.locator("#heat_drawer_btn").get_attribute(
            "aria-expanded") == "false"
        assert not page.locator("#heat_drawer_body").is_visible()
        page.locator("#heat_drawer_btn").click()
        assert page.locator("#heat_drawer_btn").get_attribute(
            "aria-expanded") == "true"
        assert page.locator("#heat_drawer_body").is_visible()
        assert page.locator("#h_nn_futurepos_8").count() == 0
        assert page.locator("#h_nn_futurepos_32").count() == 0
        assert page.locator("#h_nn_optimistic_policy").is_visible()
        assert page.locator("#h_mcts_play_selection").is_visible()
        assert page.locator("#h_mcts_visits").count() == 0
        assert page.locator("#heat_more_btn").count() == 0
        page.locator(
            '.pin-btn[data-target="h_mcts_play_selection"]').click()
        assert page.locator(
            '.pin-btn[data-target="h_mcts_play_selection"]').get_attribute(
                "aria-pressed") == "true"

        page.locator("#model_trigger").click()
        assert page.locator("#model_menu").is_visible()
        assert not page.locator("#think_pop").is_visible()
        trigger_display = page.locator("#model_trigger").evaluate(
            "(el) => getComputedStyle(el).display")
        assert trigger_display == "grid"
        trigger_label = page.locator("#model_trigger_label").bounding_box()
        trigger_caret = page.locator("#model_trigger .cs-caret").bounding_box()
        assert trigger_label and trigger_caret
        assert trigger_label["x"] + trigger_label["width"] <= trigger_caret["x"]
        page.locator("#think_trigger").click()
        assert page.locator("#think_pop").is_visible()
        assert not page.locator("#model_menu").is_visible()
        page.locator("#model_trigger").click()
        options = page.locator("#model_menu .cs-option")
        assert options.count() == 6
        trigger_box = page.locator("#model_trigger").bounding_box()
        menu_box = page.locator("#model_menu").bounding_box()
        assert trigger_box and menu_box
        assert abs(
            trigger_box["x"] + trigger_box["width"] / 2
            - menu_box["x"] - menu_box["width"] / 2
        ) < 1
        name_lefts = []
        elo_rights = []
        for i in range(options.count()):
            name_box = options.nth(i).locator(".cs-opt-name").bounding_box()
            elo_box = options.nth(i).locator(".cs-opt-elo").bounding_box()
            assert name_box and elo_box
            assert name_box["x"] + name_box["width"] < elo_box["x"]
            name_lefts.append(name_box["x"])
            elo_rights.append(elo_box["x"] + elo_box["width"])
        assert max(name_lefts) - min(name_lefts) < 1
        assert max(elo_rights) - min(elo_rights) < 1
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

        # Single-panel play_web-style settings dialog with only the current
        # controls. V7.19 Web exposes only its trained rule families.
        page.locator("#settings_btn").click()
        assert page.locator("#settings_pop").is_visible()
        assert page.locator("#settings_pop").get_attribute("role") == "dialog"
        assert page.locator(".settings-nav").count() == 0
        assert page.locator(".settings-item").count() == 5
        assert page.locator("#rule_standard").count() == 0
        assert page.locator(".seg-btn[data-rule]").count() == 2
        human_overlay = page.locator("#show_analysis_human_input")
        ai_overlay = page.locator("#show_analysis_ai_input")
        assert not human_overlay.is_checked()
        assert ai_overlay.is_checked()
        assert page.locator("#show_analysis_input").count() == 0
        human_overlay.evaluate("(input) => input.click()")
        assert page.evaluate(
            "localStorage.getItem('skz_show_analysis_human')") == "1"
        human_overlay.evaluate("(input) => input.click()")

        # V7.19 runtime board range and the two trained rule buttons.
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
            assert page.evaluate(
                """([n]) => {
                  if (coordLabel(0, 0) !== "A1"
                      || coordLabel(n - 1, n - 1)
                        !== String.fromCharCode(64 + n) + n) return false;
                  const board = document.getElementById("board");
                  const rect = board.getBoundingClientRect();
                  const hoverAt = (r, c) => {
                    board.dispatchEvent(new MouseEvent("mousemove", {
                      bubbles: true,
                      clientX: rect.left + MARGIN + c * CELL,
                      clientY: rect.top + MARGIN + r * CELL,
                    }));
                    return hoverCell && hoverCell.r === r && hoverCell.c === c;
                  };
                  return hoverAt(0, 0) && hoverAt(n - 1, n - 1);
                }""",
                [size],
            )

        for rule in ("renju", "freestyle"):
            if not page.locator("#settings_pop").is_visible():
                page.locator("#settings_btn").click()
            page.locator(f'.seg-btn[data-rule="{rule}"]').click()
            assert page.locator(
                f'.seg-btn[data-rule="{rule}"]').get_attribute(
                    "aria-pressed") == "true"

        page.locator("#settings_close_btn").click()
        assert not page.locator("#settings_pop").is_visible()
        assert "settings-open" not in (
            page.locator("body").get_attribute("class") or "")

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
        page.set_viewport_size({"width": 430, "height": 932})
        page.wait_for_timeout(250)
        board_box = page.locator("#board").bounding_box()
        board_card_box = page.locator(".board-card").bounding_box()
        chart_card_box = page.locator(".chart-card").bounding_box()
        stats_card_box = page.locator(".stats-card").bounding_box()
        assert board_box["width"] >= 414
        assert abs(board_card_box["x"] - 4) < 1
        assert abs(board_card_box["x"] - stats_card_box["x"]) < 1
        assert abs(board_card_box["width"] - stats_card_box["width"]) < 1
        board_insets = (
            board_box["x"] - board_card_box["x"],
            board_box["y"] - board_card_box["y"],
            board_card_box["x"] + board_card_box["width"]
            - board_box["x"] - board_box["width"],
            board_card_box["y"] + board_card_box["height"]
            - board_box["y"] - board_box["height"],
        )
        assert max(board_insets) - min(board_insets) < 1
        radii = page.locator(".board-card").evaluate(
            """(card) => ({
              outer: parseFloat(getComputedStyle(card).borderRadius),
              inner: parseFloat(getComputedStyle(
                document.getElementById("board")).borderRadius),
            })"""
        )
        assert abs(
            radii["outer"] - radii["inner"] - board_insets[0]
        ) < 1
        board_to_chart = (
            chart_card_box["y"] - board_card_box["y"]
            - board_card_box["height"])
        chart_to_stats = (
            stats_card_box["y"] - chart_card_box["y"]
            - chart_card_box["height"])
        assert abs(board_to_chart - chart_to_stats) < 1
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
        assert tree_page.locator("#rule option").evaluate_all(
            "(options) => options.map((option) => option.value)"
        ) == ["renju", "freestyle"]
        tree_page.locator("#btnStep").click()
        tree_page.wait_for_function(
            """() => document.getElementById("status")
              .textContent.startsWith("1 次模拟")""",
            timeout=120_000,
        )

        direct_page = browser.new_page(
            viewport={"width": 1440, "height": 1000})
        direct_page.add_init_script(
            """window.__difficultyEverVisible = false;
               document.addEventListener("DOMContentLoaded", () => {
                 const sample = () => {
                   const modal =
                     document.getElementById("difficulty_modal");
                   if (modal && getComputedStyle(modal).display !== "none")
                     window.__difficultyEverVisible = true;
                   requestAnimationFrame(sample);
                 };
                 sample();
               });""")
        direct_page.goto(
            f"{BASE_URL}/gomoku/lv1",
            wait_until="domcontentloaded",
            timeout=120_000,
        )
        wait_ready(direct_page)
        assert direct_page.locator(
            "#model_trigger_label").inner_text() == "LV1 入门"
        assert not direct_page.evaluate("window.__difficultyEverVisible")
        assert "LV1 入门" in direct_page.locator(
            "#loading_text").inner_text()

        # Session creation has no real progress API. Verify that its simulated
        # progress advances but remains below completion, and that a slow
        # WeChat initialization gets an actionable compatibility hint.
        direct_page.evaluate(
            """() => {
              showLoadingOverlay("loading_model", "LV1", "入门");
              startInitializationFeedback("LV1", "入门");
            }"""
        )
        direct_page.wait_for_function(
            """() => parseFloat(
              document.getElementById("loading_fill").style.width) > 4"""
        )
        init_pct = direct_page.locator("#loading_pct").inner_text()
        assert init_pct.endswith("%")
        assert float(init_pct[:-1]) <= 92
        direct_page.evaluate("showInitializationSlowHint(true)")
        assert "微信内初始化时间过长" in direct_page.locator(
            "#loading_text").inner_text()
        direct_page.evaluate("hideLoadingOverlay()")

        error_context = browser.new_context()
        error_page = error_context.new_page()
        error_page.route(
            "**/models/level1.onnx*",
            lambda route: route.fulfill(
                status=503, content_type="text/plain", body="offline"),
        )
        error_page.goto(
            f"{BASE_URL}/gomoku/lv1",
            wait_until="domcontentloaded",
            timeout=120_000,
        )
        error_page.locator("#loading_text").filter(
            has_text="错误:").wait_for(timeout=120_000)
        assert error_page.locator(
            "#loading_pct").inner_text() == "请刷新页面后重试"
        error_context.close()

        browser.close()

    if errors:
        raise AssertionError("\n".join(errors))
    print("browser-v719-smoke: ok")


if __name__ == "__main__":
    main()
