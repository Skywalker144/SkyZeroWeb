from playwright.sync_api import sync_playwright
import json
import argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1920,'height':1080})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:print('CONSOLE',m.type,m.text,flush=True) if m.type=='error' else None)
    page.add_init_script("""localStorage.setItem('skz_theme','dark');window.downloadProgress=[];
      new MutationObserver(records=>{for(const r of records){if(r.target.id==='model-progress')window.downloadProgress.push(Number(r.target.value));}}).observe(document,{subtree:true,attributes:true,attributeFilter:['value']});""")
    page.goto(args.url.rstrip('/')+'/scotlandyard/')
    assert page.locator('#setup-panel, .advanced-card, #quick-ruleset, #quick-evader, #quick-downloads').count()==0
    page.wait_for_function("document.querySelector('#app').classList.contains('has-game')",timeout=120000)
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    assert not page.locator('#error').is_visible()
    assert page.locator('#model-loading').is_hidden()
    assert page.locator('#control-title').inner_text()=='对局控制'
    assert page.locator('#open-setup, #settings-dialog, #theme-select').count()==0
    assert page.locator('#match-pursuer').inner_text()=='AZ Pursuer'
    assert page.evaluate("document.documentElement.dataset.theme === 'dark'")
    assert page.locator('#quick-pursuer option').all_text_contents()==['AZ','PPO GNN']
    assert page.evaluate("""() => {
      const strategies=document.querySelector('.quick-strategies').getBoundingClientRect();
      const actions=document.querySelector('.match-actions').getBoundingClientRect();
      return actions.top-strategies.bottom >= 12;
    }""")
    assert page.evaluate('downloadProgress.some(v => v > 0)')
    print('STARTED',page.locator('#error').inner_text(),flush=True)
    page.wait_for_timeout(350)
    page.screenshot(path='/tmp/scotland-game.png')
    for i in range(3):
        if not page.locator('.move-btn').count(): break
        page.locator('.move-btn').first.click()
        page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
        print('MOVE',i,page.locator('#turn-label').inner_text(),page.locator('#error').inner_text(),flush=True)
    page.locator('#undo').click()
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'")
    print('UNDO',page.locator('#turn-label').inner_text(),flush=True)
    page.reload()
    page.wait_for_function("document.querySelector('#app').classList.contains('has-game')",timeout=120000)
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    print('RESTORED',page.locator('#turn-label').inner_text(),flush=True)
    previous_game=json.loads(page.evaluate("sessionStorage.getItem('skz_scotland_game_v1')"))
    page.locator('#quick-pursuer').select_option('ppo-gnn-000500')
    page.locator('#model-loading:not([hidden])').wait_for(timeout=10000)
    assert page.locator('#loading-title').inner_text()=='正在下载 PPO GNN 权重'
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    switched_game=json.loads(page.evaluate("sessionStorage.getItem('skz_scotland_game_v1')"))
    assert switched_game['id'] != previous_game['id']
    assert switched_game['metadata']['seed'] == previous_game['metadata']['seed']
    assert switched_game['model']['id']=='ppo-gnn-000500'
    assert switched_game['game']['state']['round']==1
    assert page.locator('#match-pursuer').inner_text()=='PPO Pursuer'
    print('PPO',page.locator('#match-pursuer').inner_text(),page.locator('#error').inner_text(),flush=True)
    page.locator('.move-btn').first.click()
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    assert not page.locator('#error').is_visible()
    print('PPO MOVE',page.locator('#turn-label').inner_text(),page.locator('#error').inner_text(),flush=True)
    page.locator('#quick-pursuer').select_option('az-selfplay-000360')
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    assert page.locator('#match-pursuer').inner_text()=='AZ Pursuer'
    assert json.loads(page.evaluate("sessionStorage.getItem('skz_scotland_game_v1')"))['game']['state']['round']==1
    # Verify browser WASM actions against the original PyTorch regression fixtures.
    catalog=json.loads((root/'public/scotlandyard/models.json').read_text())
    for model in catalog['pursuer']:
        fixtures=json.loads((root/f"tests/{model['id']}-fixtures.json").read_text())
        result=page.evaluate("""async ({model,fixtures}) => {
          const worker=new Worker(new URL('./inference-worker.js',location.href));
          let id=0;
          const call=message=>new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>reject(new Error('fixture timeout')),60000);
            worker.onmessage=({data})=>{if(data.progress)return;clearTimeout(timer);data.error?reject(new Error(data.error)):resolve(data.result);};
            worker.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};
            worker.postMessage({id:++id,...message});
          });
          try {
            await call({kind:'load',model,url:new URL('./models/'+model.id+'.onnx',location.href).href});
            const outputs=[];
            for (const fixture of fixtures) outputs.push(await call({kind:'run',inputs:Object.fromEntries(Object.entries(fixture.inputs).map(([key,v])=>[key,v.data]))}));
            return outputs;
          } finally {worker.terminate();}
        }""",{'model':model,'fixtures':fixtures})
        for actual,fixture in zip(result,fixtures):
            assert actual['actions']==fixture['actions']
            assert actual['order']==fixture['order']
        print('WASM PARITY',model['id'],len(fixtures),flush=True)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path='/tmp/scotland-mobile.png',full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors,errors
    print('PASS automatic start, real download progress, mobile overflow, WASM/native parity',flush=True)
    # A failed model download must leave Start usable, and retry must recover.
    context=browser.new_context()
    context.route('**/scotlandyard/models/*.onnx',lambda route:route.abort())
    failed=context.new_page()
    failed.goto(args.url.rstrip('/')+'/scotlandyard/')
    failed.locator('#error:not([hidden])').wait_for(timeout=60000)
    failed.locator('#loading-retry:not([hidden])').wait_for()
    assert not failed.locator('#app').evaluate("el=>el.classList.contains('has-game')")
    context.unroute('**/scotlandyard/models/*.onnx')
    failed.locator('#loading-retry').click()
    failed.wait_for_function("document.querySelector('#app').classList.contains('has-game')",timeout=120000)
    failed.wait_for_function("document.querySelector('#app').getAttribute('aria-busy') === 'false'",timeout=120000)
    assert not failed.locator('#error').is_visible()
    print('PASS failed model download and retry recovery',flush=True)
    context.close()
    browser.close()
