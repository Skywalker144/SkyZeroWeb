import os

from playwright.sync_api import sync_playwright


base_url = os.environ.get(
    "SKYZERO_WEB_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
errors = []

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 430, "height": 900})
    page.on("console", lambda message: errors.append(
        f"console {message.type}: {message.text}")
        if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    page.add_init_script("""(() => {
      localStorage.setItem('skz_name', 'Browser Test');
      localStorage.setItem('skz_token', 'browser-test-token');
    })()""")
    page.route("**/api/**", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"top":[]}'))
    page.goto(f"{base_url}/2048", wait_until="domcontentloaded")
    page.locator("#ai_auto_btn").wait_for(state="visible")
    page.wait_for_function(
        "() => !document.getElementById('ai_auto_btn').disabled",
        timeout=120_000,
    )
    before = page.locator("#tiles").inner_html()
    page.locator("#ai_auto_btn").click()
    assert page.locator("#ai_auto_btn").get_attribute("aria-pressed") == "true"
    page.wait_for_function(
        "before => document.getElementById('tiles').innerHTML !== before",
        arg=before,
        timeout=120_000,
    )
    page.locator("#ai_auto_btn").click()
    assert page.locator("#ai_auto_btn").get_attribute("aria-pressed") == "false"
    assert not errors, "\n".join(errors)
    browser.close()
