import json
from pathlib import Path
import os
import numpy as np
from playwright.sync_api import sync_playwright
from browser_runtime import use_cached_runtime, select_control

ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('SKYZERO_WEB_BASE_URL','http://127.0.0.1:8000').rstrip('/')

def wait_ready(page):
    page.wait_for_function('typeof session !== "undefined" && session.ready && !session.busy && !session.thinking && !session.error && session.state !== null',timeout=120_000)


def probe(page,request):
    return page.evaluate('''request=>new Promise((resolve,reject)=>{
        const id='probe-'+Math.random();
        const listener=({data})=>{
            if(data.id!==id)return;
            session.worker.removeEventListener('message',listener);
            if(data.type==='error')reject(new Error(data.message));
            else resolve(data.state || {...data.result,policy:Array.from(data.result.policy),wdl:Array.from(data.result.wdl)});
        };
        session.worker.addEventListener('message',listener);session.worker.postMessage({id,...request});
    })''',request)

with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport=dict(width=1440,height=1000))
    use_cached_runtime(page.context)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+'/go/',wait_until='domcontentloaded')
    wait_ready(page)
    assert page.locator('#size').input_value()=='19'
    assert page.locator('.go-point').count()==361
    fixture=json.loads((ROOT/'tests/go-rules.json').read_text())
    for row in fixture['cases']:
        actual=probe(page,dict(type='position',size=row['size'],actions=row['actions']))
        for key in ['board','legal','player','finished','winner','score']:assert actual[key]==row[key],(row['size'],len(row['actions']),key)
    fixture=json.loads((ROOT/'tests/go-inference.json').read_text())
    for row in fixture['cases']:
        actual=probe(page,dict(type='evaluate',size=row['size'],actions=row['actions']))
        for name,expected in row['outputs'].items():np.testing.assert_allclose(actual['raw'][name],expected,rtol=3e-4,atol=3e-4)
        assert abs(sum(actual['policy'])-1)<1e-5
        state=probe(page,dict(type='position',size=row['size'],actions=row['actions']))
        assert all(v==0 for a,v in enumerate(actual['policy']) if a not in state['legal'])
    print('Go native rules and browser ONNX probes passed',flush=True)
    select_control(page, '#budget','500')
    page.locator('[data-action="60"]').focus();page.keyboard.press('Enter')
    page.wait_for_function('session.actions.length===2 && !session.thinking && !session.busy',timeout=60_000)
    assert page.locator('.go-stone').count()==2
    assert page.locator('.candidate-row').count()>0
    page.locator('#undo').click();wait_ready(page)
    assert page.evaluate('session.actions.length')==0
    page.locator('#pass').click()
    page.wait_for_function('session.actions.length===2 && !session.thinking && !session.busy',timeout=60_000)
    assert page.evaluate('session.actions[0]')==361
    page.locator('#new-game').click();wait_ready(page)
    select_control(page, '#budget','5000')
    page.locator('[data-action="60"]').click();page.wait_for_function('session.thinking')
    page.locator('#new-game').click();wait_ready(page)
    page.wait_for_timeout(800)
    assert page.evaluate('session.actions.length')==0
    select_control(page, '#budget','500')
    select_control(page, '#side','-1');wait_ready(page)
    assert page.evaluate('session.actions.length===1 && session.state.player===-1')
    select_control(page, '#side','1');wait_ready(page)
    page.locator('#settings_btn').click()
    assert page.locator('#settings_pop').is_visible()
    page.locator('#size + .custom-select .cs-trigger').click()
    page.locator('#size + .custom-select .cs-option').filter(has_text='9 × 9').click()
    page.keyboard.press('Escape');wait_ready(page)
    assert page.locator('.go-point').count()==81
    select_control(page, '#size','13');wait_ready(page)
    assert page.locator('.go-point').count()==169
    select_control(page, '#size','19');wait_ready(page)
    page.locator('#language').click()
    assert page.locator('#status').inner_text()=='Your turn'
    select_control(page, '#theme','dark')
    assert page.evaluate('document.documentElement.dataset.theme')=='dark'
    page.screenshot(path='/tmp/go-desktop.png',full_page=True)
    for width in [390,320]:
        page.set_viewport_size(dict(width=width,height=844))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('#settings_btn').click()
        assert page.locator('#settings_pop').is_visible()
        page.keyboard.press('Escape')
    page.screenshot(path='/tmp/go-mobile.png',full_page=True)
    page.goto(BASE+'/',wait_until='domcontentloaded')
    page.locator('a[href="/go/"]').click();wait_ready(page)
    assert not errors,errors
    browser.close()
print('browser-go-smoke: ok')
