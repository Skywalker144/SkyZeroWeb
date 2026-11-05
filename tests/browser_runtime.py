import os
from pathlib import Path


def use_cached_runtime(context):
    directory = os.environ.get('SKYZERO_ORT_DIR')
    if directory:
        root = Path(directory)
        def serve(route):
            path = root / route.request.url.rsplit('/', 1)[-1]
            if not path.is_file():
                raise FileNotFoundError(path)
            route.fulfill(path=str(path), content_type='application/wasm' if path.suffix == '.wasm' else 'text/javascript',
                          headers={'Access-Control-Allow-Origin': '*'})
        context.route('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/**', serve)


def select_control(page, selector, value):
    source=page.locator(selector)
    label=source.evaluate('(select,value)=>Array.from(select.options).find(option=>option.value===value).text',value)
    settings=source.evaluate('(select)=>!!select.closest("#settings_pop")')
    opened=settings and not page.locator('#settings_pop').is_visible()
    if opened:page.locator('#settings_btn').click()
    control=page.locator(selector+' + div')
    if source.get_attribute('data-board-segment') is not None:
        control.get_by_role('button',name=label,exact=True).click()
    else:
        control.locator('.cs-trigger').click()
        control.get_by_role('option',name=label,exact=True).click()
    if opened:page.keyboard.press('Escape')


def click_settings_action(page, selector):
    opened=not page.locator('#settings_pop').is_visible()
    if opened:page.locator('#settings_btn').click()
    page.locator(selector).click()
    if opened:page.keyboard.press('Escape')
