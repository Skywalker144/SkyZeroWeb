from browser_runtime import use_cached_runtime, select_control, click_settings_action
import json
import os
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


root = Path(__file__).resolve().parents[1]
base = os.environ.get('SKYZERO_WEB_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
errors = []


def wait_ready(page):
    page.wait_for_function("() => ready && !thinking && !balancing && !error", timeout=120_000)


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1200, 'height': 950})
    use_cached_runtime(page.context)
    page.on('pageerror', lambda problem: errors.append(str(problem)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.goto(base + '/hex/', wait_until='domcontentloaded')
    wait_ready(page)
    expected_manifest = json.loads((root / 'public/hex/models/manifest.json').read_text())
    assert page.evaluate('manifest') == expected_manifest
    assert page.locator('.cell').count() == 225
    assert page.locator('.board-panel.card, .board-panel .board-wrap').count() == 0
    assert page.locator('.board-panel').evaluate('(node)=>getComputedStyle(node).backgroundColor') == 'rgba(0, 0, 0, 0)'
    assert page.locator('.cell').nth(0).evaluate('(node)=>node.getBBox().y') == page.locator('.cell').nth(1).evaluate('(node)=>node.getBBox().y')
    assert page.locator('#model option').count() == 3
    for model in [item['id'] for item in expected_manifest['models']]:
        select_control(page, '#model', model)
        wait_ready(page)
        fixture = json.loads((root / 'tests' / f'hex-{model}-inference.json').read_text())
        artifact = page.evaluate("""async () => {
            const model = manifest.models.find(item => item.id === document.getElementById('model').value);
            const response = await fetch('/hex/models/' + model.file);
            const bytes = await response.arrayBuffer();
            const digest = await crypto.subtle.digest('SHA-256', bytes);
            return {bytes: bytes.byteLength, sha256: Array.from(new Uint8Array(digest),
                value => value.toString(16).padStart(2, '0')).join('')};
        }""")
        expected_artifact = next(item for item in expected_manifest['models'] if item['id'] == model)
        assert artifact['bytes'] == expected_artifact['bytes']
        assert artifact['sha256'] == expected_artifact['sha256']
        for item in fixture['cases']:
            actual = page.evaluate("""item => new Promise((resolve, reject) => {
                const id = 'probe-' + Math.random();
                const listener = ({data}) => {
                    if (data.id !== id) return;
                    worker.removeEventListener('message', listener);
                    if (data.type === 'error') reject(new Error(data.message));
                    else resolve(data.result);
                };
                worker.addEventListener('message', listener);
                worker.postMessage({id, type: 'evaluate', ...item});
            })""", {key: item[key] for key in ['board', 'player', 'size', 'rotation']})
            for name, expected in item['outputs'].items():
                np.testing.assert_allclose(actual['raw'][name], np.array(expected).flatten(), rtol=2e-4, atol=2e-4)
            for action, stone in enumerate(item['board']):
                if stone:
                    assert actual['policy'][action] == 0
            assert abs(sum(actual['policy']) - 1) < 1e-5
        print(f'{model}: browser WASM parity on {len(fixture["cases"])} positions', flush=True)
        select_control(page, '#budget', '500')
        page.locator('[data-action="112"]').click()
        page.wait_for_function('() => history.length === 2 && !thinking', timeout=60_000)
        assert page.locator('.cell.black').count() == 1
        assert page.locator('circle.hex-piece.black').count() == 1
        assert page.locator('.cell.white').count() == 1
        assert page.locator('circle.hex-piece.white').count() == 1
        assert page.locator('.candidate-row').count() > 0
        assert page.locator('#evaluation-value').inner_text().endswith('%')
        page.locator('#undo').click()
        assert page.evaluate('history.length') == 0
        assert page.locator('#evaluation-value').inner_text() == '—'
        select_control(page, '#side', '-1')
        click_settings_action(page, '#balanced-opening')
        page.wait_for_function('() => opening && !balancing && !thinking && history.length === 1', timeout=120_000)
        assert page.locator('.cell.black').count() == 1
        assert page.locator('circle.hex-piece.black').count() == 1
        assert page.locator('.cell.white').count() == 0
        assert page.locator('circle.hex-piece.white').count() == 0
        assert page.locator('#undo').is_disabled()
        assert page.evaluate('opening.attempts >= 1 && opening.whiteWinProbability >= 0 && opening.whiteWinProbability <= 1')
        action = page.evaluate('Array.from(board).indexOf(0)')
        page.locator(f'[data-action="{action}"]').click()
        page.wait_for_function('() => history.length === 3 && !thinking', timeout=60_000)
        page.locator('#undo').click()
        assert page.evaluate('history.length === 1 && board[opening.action] === 1')
        print(f'{model}: balanced opening and protected-prefix undo OK', flush=True)
        select_control(page, '#side', '1')
    page.evaluate("""() => {
        document.getElementById('balanced-opening').click();
        if (!balancing) throw new Error('Opening did not start');
        document.getElementById('balanced-opening').click();
    }""")
    page.wait_for_timeout(1000)
    assert page.evaluate('!balancing && !opening && history.length === 0')
    click_settings_action(page, '#balanced-opening')
    page.wait_for_function('() => opening && !balancing && !thinking && history.length === 2', timeout=120_000)
    assert page.locator('#undo').is_disabled()
    action = page.evaluate('Array.from(board).indexOf(0)')
    page.locator(f'[data-action="{action}"]').click()
    page.wait_for_function('() => history.length === 4 && !thinking', timeout=60_000)
    page.locator('#undo').click()
    assert page.evaluate('history.length === 2 && board[opening.action] === 1')
    page.locator('#new-game').click()
    assert page.evaluate('!opening && history.length === 0')
    page.evaluate("""() => {
        document.getElementById('balanced-opening').click();
        document.getElementById('model').value = manifest.models[0].id;
        document.getElementById('model').dispatchEvent(new Event('change'));
    }""")
    wait_ready(page)
    assert page.evaluate('!opening && history.length === 0')
    page.locator('[data-action="112"]').click()
    page.locator('#new-game').click()
    page.wait_for_timeout(1500)
    assert page.evaluate('history.length') == 0
    select_control(page, '#size', '9')
    select_control(page, '#side', '-1')
    page.wait_for_function('() => history.length === 1 && !thinking', timeout=60_000)
    assert page.locator('.cell').count() == 81
    action = page.evaluate('Array.from(board).indexOf(0)')
    page.locator(f'[data-action="{action}"]').click()
    page.wait_for_function('() => history.length === 3 && !thinking', timeout=60_000)
    page.locator('#undo').click()
    assert page.evaluate('history.length') == 1
    select_control(page, '#side', '1')
    select_control(page, '#size', '9')
    page.evaluate("""() => {
        history = [0, 8, 9, 17, 18, 26, 27, 35, 36, 44, 45, 53, 54, 62, 63, 71];
        board = new Int8Array(81);
        history.forEach((action, i) => { board[action] = i % 2 ? -1 : 1; });
        renderBoard();
    }""")
    page.locator('[data-action="72"]').focus()
    page.keyboard.press('Enter')
    assert page.locator('#status').inner_text() == '你赢了！'
    assert page.locator('.cell.connected').count() == 9
    assert page.evaluate('!thinking && history.length === 17')
    page.locator('#new-game').click()
    page.locator('#language').click()
    assert page.locator('#status').inner_text() == 'Your turn'
    select_control(page, '#theme', 'dark')
    page.wait_for_timeout(200)
    assert page.evaluate('document.documentElement.dataset.theme') == 'dark'
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    (root / '.cache').mkdir(exist_ok=True)
    page.screenshot(path=str(root / '.cache/hex-mobile.png'), full_page=True)
    page.set_viewport_size({'width': 1200, 'height': 950})
    select_control(page, '#size', '15')
    select_control(page, '#theme', 'light')
    page.wait_for_timeout(200)
    page.screenshot(path=str(root / '.cache/hex-desktop.png'), full_page=True)
    page.goto(base + '/', wait_until='domcontentloaded')
    assert page.locator('a[href="/hex/"]').count() == 1
    failure = browser.new_page()
    use_cached_runtime(failure.context)
    failure.route('**/hex/models/*.onnx', lambda route: route.fulfill(status=503, body='offline'))
    failure.goto(base + '/hex/', wait_until='domcontentloaded')
    failure.wait_for_function('() => error.includes("503")')
    assert failure.locator('#board').get_attribute('aria-disabled') == 'true'
    failure.unroute('**/hex/models/*.onnx')
    failure.locator('#new-game').click()
    wait_ready(failure)
    failure.close()
    failure = browser.new_page()
    use_cached_runtime(failure.context)
    failure.route('**/hex/models/manifest.json*', lambda route: route.fulfill(status=503, body='offline'))
    failure.goto(base + '/hex/', wait_until='domcontentloaded')
    failure.wait_for_function('() => error.includes("503")')
    failure.unroute('**/hex/models/manifest.json*')
    failure.locator('#new-game').click()
    wait_ready(failure)
    failure.close()
    browser.close()
    assert not errors, '\n'.join(errors)
print('browser-hex-smoke: ok')
