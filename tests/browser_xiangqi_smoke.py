import json
import os
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get('SKYZERO_WEB_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
errors = []


def wait_ready(page):
    page.wait_for_function('ready && !busy && !thinking && !error && state !== null', timeout=120_000)


def probe(page, request):
    return page.evaluate('''request => new Promise((resolve, reject) => {
        const id = 'probe-' + Math.random();
        const listener = ({data}) => {
            if (data.id !== id) return;
            worker.removeEventListener('message', listener);
            if (data.type === 'error') reject(new Error(data.message));
            else resolve(data.state || {...data.result, policy: Array.from(data.result.policy), wdl: Array.from(data.result.wdl)});
        };
        worker.addEventListener('message', listener);
        worker.postMessage({id, ...request});
    })''', request)


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport=dict(width=1200, height=1000))
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.goto(BASE + '/xiangqi', wait_until='domcontentloaded')
    wait_ready(page)
    assert page.locator('.piece-name').count() == 32
    assert page.locator('.square').count() == 90
    manifest = json.loads((ROOT / 'public/xiangqi/models/manifest.json').read_text())
    assert page.evaluate('manifest') == manifest
    fixture = json.loads((ROOT / 'tests/xiangqi-inference.json').read_text())
    for item in fixture['cases']:
        actual = probe(page, dict(type='evaluate', actions=item['actions'], mirror=item['mirror']))
        for name, expected in item['outputs'].items():
            np.testing.assert_allclose(actual['raw'][name], expected, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(actual['policy'], item['policy'], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(actual['wdl'], item['wdl'], rtol=2e-4, atol=2e-4)
        assert abs(sum(actual['policy']) - 1) < 1e-5
        state = probe(page, dict(type='position', actions=item['actions']))
        assert all(value == 0 for action, value in enumerate(actual['policy']) if action not in state['legal'])
    print(f'{len(fixture["cases"])} browser WASM inference probes match native ONNX', flush=True)
    rules = json.loads((ROOT / 'tests/xiangqi-rules.json').read_text())
    for item in rules['cases']:
        actual = probe(page, dict(type='position', fen=item['fen'], actions=item['actions']))
        for key in ['board', 'legal', 'player', 'winner', 'finished', 'reason']:
            assert actual[key] == item['states'][-1][key], (item['name'], key)
    artifact = page.evaluate('''async () => {
        const bytes = await (await fetch('/xiangqi/models/' + manifest.model.file)).arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        return {bytes: bytes.byteLength, sha256: Array.from(new Uint8Array(digest),
            value => value.toString(16).padStart(2, '0')).join('')};
    }''')
    assert artifact['bytes'] == manifest['model']['bytes']
    assert artifact['sha256'] == manifest['model']['sha256']
    page.select_option('#budget', '500')
    page.locator('[data-square="30"]').focus()
    page.keyboard.press('Enter')
    assert page.locator('.destination').count() == 1
    page.locator('[data-square="40"]').focus()
    page.keyboard.press('Enter')
    page.wait_for_function('actions.length === 2 && !thinking && !busy', timeout=60_000)
    assert page.evaluate('state.board[40]') == 4
    assert page.evaluate('state.player') == 1
    assert page.locator('#last-move').inner_text() != '—'
    assert page.locator('.candidate-row').count() > 0
    assert page.locator('#evaluation-value').inner_text().endswith('%')
    page.locator('#undo').click()
    wait_ready(page)
    assert page.evaluate('actions.length') == 0
    assert page.locator('#evaluation-value').inner_text() == '—'
    page.select_option('#budget', '5000')
    page.locator('[data-square="30"]').click()
    page.locator('[data-square="40"]').click()
    page.wait_for_function('thinking')
    page.wait_for_function("() => /\\d+\\.\\d s/.test(document.querySelector('#search-info').textContent)")
    assert 'NaN' not in page.locator('#search-info').inner_text()
    page.locator('#undo').click()
    wait_ready(page)
    page.wait_for_timeout(800)
    assert page.evaluate('actions.length === 0 && state.player === 1')
    page.locator('[data-square="30"]').click()
    page.locator('[data-square="40"]').click()
    page.wait_for_function('thinking')
    page.locator('#new-game').click()
    wait_ready(page)
    page.wait_for_timeout(800)
    assert page.evaluate('actions.length') == 0
    page.select_option('#budget', '500')
    page.select_option('#side', '-1')
    wait_ready(page)
    assert page.evaluate('actions.length === 1 && state.player === -1')
    assert page.locator('#undo').is_disabled()
    move = page.evaluate('moves[state.legal[0]]')
    page.locator(f'[data-square="{move[1]*10+move[0]}"]').click()
    page.locator(f'[data-square="{move[3]*10+move[2]}"]').click()
    wait_ready(page)
    assert page.evaluate('actions.length') == 3
    page.locator('#undo').click()
    wait_ready(page)
    assert page.evaluate('actions.length === 1 && state.player === -1')
    page.select_option('#side', '1')
    wait_ready(page)
    page.locator('#language').click()
    assert page.locator('#status').inner_text() == 'Your turn'
    page.select_option('#theme', 'dark')
    assert page.evaluate('document.documentElement.dataset.theme') == 'dark'
    screenshot_dir = ROOT / '.cache'
    screenshot_dir.mkdir(exist_ok=True)
    page.screenshot(path=str(screenshot_dir / 'xiangqi-desktop.png'), full_page=True)
    page.set_viewport_size(dict(width=390, height=844))
    page.locator('#language').click()
    page.select_option('#theme', 'light')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path=str(screenshot_dir / 'xiangqi-mobile.png'), full_page=True)
    page.goto(BASE + '/', wait_until='domcontentloaded')
    page.locator('a[href="/xiangqi/"]').click()
    wait_ready(page)
    assert errors == [], errors
    browser.close()
    print('Rules, moves, cancellation, undo, both sides, home entry and mobile layout passed', flush=True)
