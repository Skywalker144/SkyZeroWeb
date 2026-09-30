import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

base_url = os.environ.get('SKYZERO_WEB_BASE_URL', 'http://127.0.0.1:8788').rstrip('/')
artifact_dir = Path(os.environ.get('POKER_SCREENSHOT_DIR', '/tmp'))
errors = []
requests = []


def decision(route):
    body = route.request.post_data_json
    requests.append(body)
    observation = body['observation']
    action = 'k' if observation['legal']['check'] else 'c'
    chunks = [
        {'pokerskill': {'engine': 'author', 'bigBlind': 2, 'analysisMs': 5,
                       'prompt': 'Private strategy for browser validation',
                       'version': 'browser-test', 'street': observation['street']}},
        {'choices': [{'delta': {'content': json.dumps({'action': action,
                                                     'reasoning': '根据公共信息控制底池。'})}}]},
        {'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
    ]
    stream = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
    route.fulfill(status=200, content_type='text/event-stream', body=stream + 'data: [DONE]\n\n')


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/api/poker/decision', decision)
    page.goto(f'{base_url}/poker/', wait_until='networkidle')
    assert page.locator('header, .home-link, #mode, #slider').count() == 0
    assert page.locator('#key').is_visible()
    page.locator('#newhand').click()
    assert 'API key' in page.locator('#error').inner_text()
    page.locator('#key').fill('browser-test-key')
    page.locator('#newhand').click()
    assert page.locator('#newhand').is_hidden()
    assert page.locator('#settings').get_attribute('open') is None
    assert '再投入 3' in page.locator('#raise-preview').inner_text()
    page.locator('#amount').fill('3')
    assert page.locator('#raise').is_disabled()
    page.locator('[data-size=pot]').click()
    assert page.locator('#amount').input_value() == '6'
    assert '再投入 5' in page.locator('#raise-preview').inner_text()
    page.locator('#amount').fill('5.5')
    assert page.locator('#raise').is_disabled()
    page.locator('#amount').fill('6')
    page.locator('#amount').press('Enter')
    page.wait_for_function("document.querySelector('#turn').textContent === '轮到你行动'")
    assert requests[0]['observation']['legal']['call'] == 4
    assert page.locator('#history .entry.ai').count() == 2
    assert 'Private strategy' not in page.locator('#review').inner_html()
    assert 'DeepSeek · 过牌' in page.locator('#last-action').inner_text()
    page.locator('#history').evaluate('(el) => {el.scrollTop=el.scrollHeight;}')
    scroll = page.locator('#history').evaluate('(el) => el.scrollTop')
    page.locator('#settings > summary').click()
    page.locator('#settings > summary').click()
    assert page.locator('#history').evaluate('(el) => el.scrollTop') == scroll
    for width, height in [(1440, 900), (1280, 720), (1024, 768)]:
        page.set_viewport_size({'width': width, 'height': height})
        box = page.locator('.actionbox').bounding_box()
        assert box['y'] + box['height'] <= height, f'controls below fold at {width}x{height}'
    page.set_viewport_size({'width': 1440, 'height': 900})
    page.screenshot(path=str(artifact_dir / 'poker-playing-desktop.png'), full_page=True)
    page.locator('#fold').click()
    assert page.locator('#newhand').is_visible()
    assert page.locator('#raise-controls').is_hidden()
    assert 'Private strategy' in page.locator('#review').inner_html()
    assert '赢得底池' in page.locator('#history .entry').first.inner_text()
    strategy = page.locator('#review details').first
    strategy.locator('summary').click()
    page.locator('#settings > summary').click()
    page.locator('#stackmode').select_option('fixed200')
    assert strategy.get_attribute('open') is not None
    page.screenshot(path=str(artifact_dir / 'poker-review-desktop.png'), full_page=True)
    for width, height in [(1440, 900), (1280, 720), (1024, 768), (980, 800), (768, 1024), (430, 900), (375, 812), (320, 720)]:
        page.set_viewport_size({'width': width, 'height': height})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'overflow at {width}'
    page.set_viewport_size({'width': 390, 'height': 844})
    page.reload(wait_until='networkidle')
    page.locator('#newhand').click()
    assert page.locator('#key').is_visible()
    page.locator('#key').fill('browser-test-key')
    page.locator('#newhand').click()
    page.locator('#call').click()
    page.wait_for_function("document.querySelector('#turn').textContent === '轮到你行动'")
    page.locator('label[for=show-history]').click()
    assert page.locator('.history-sidebar').is_visible()
    assert page.locator('.review-panel').is_hidden()
    page.screenshot(path=str(artifact_dir / 'poker-playing-mobile.png'), full_page=True)
    page.locator('#fold').click()
    page.locator('label[for=show-review]').click()
    assert page.locator('.review-panel').is_visible()
    assert page.locator('.history-sidebar').is_hidden()
    page.locator('label[for=show-settings]').click()
    page.locator('#settings > summary').click()
    assert page.locator('#key').is_visible()
    assert not errors, '\n'.join(errors)
    print(f'Poker browser checks passed; {len(requests)} streamed decisions; desktop and mobile layouts checked.')
    browser.close()
