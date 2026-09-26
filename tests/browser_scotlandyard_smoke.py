import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8076')
parser.add_argument('--reveals-only',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
catalog=json.loads((root/'public/scotlandyard/models.json').read_text())
board=json.loads((root/'public/scotlandyard/board.json').read_text())
STORE_KEY='skz_scotland_v016_blind_20_v1'
url=args.url.rstrip('/')+'/scotlandyard/'


def data(page):
    return page.evaluate('(key)=>JSON.parse(localStorage.getItem(key))',STORE_KEY)


def state(page):
    return data(page)['active']['game']['game']['state']


def ready(page):
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy')==='false' && !document.querySelector('#evaluation-start').disabled",timeout=120000)
    assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()


def export(page,name):
    with page.expect_download() as download:
        page.locator('#evaluation-export').click()
    path=Path('/tmp')/name
    download.value.save_as(path)
    return json.loads(path.read_text())


def greedy(page):
    return page.evaluate('''async () => {
      const {requestJson}=await import('./local-api.js?v=v016-preload-1');
      const board=await requestJson('/api/map');
      const current=JSON.parse(localStorage.getItem('skz_scotland_v016_blind_20_v1')).active;
      const s=await requestJson(`/api/games/${current.id}/state`);
      const dist=new Map(s.pursuer_positions.map(n=>[n,0])),queue=[...s.pursuer_positions];
      for(let i=0;i<queue.length;i++) for(const m of board.actions[queue[i]]) {
        if(m.ticket==='black'||dist.has(m.destination)) continue;
        dist.set(m.destination,dist.get(queue[i])+1);queue.push(m.destination);
      }
      return [...s.legal_evader_moves].sort((a,b)=>(dist.get(b.destination)-dist.get(a.destination)) || Number(a.ticket==='black')-Number(b.ticket==='black'))[0];
    }''')


def play(page,move):
    page.locator(f'.destination-group[data-destination="{move["destination"]}"] .move-btn[data-ticket="{move["ticket"]}"]').click()
    ready(page)


with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1600,'height':900},reduced_motion='reduce')
    context.add_init_script('''
      if(!localStorage.getItem('skz_scotland_v016_unlimited_undo_v1')) localStorage.setItem('skz_scotland_v016_unlimited_undo_v1','old pilot results');
      const random=crypto.getRandomValues.bind(crypto);
      crypto.getRandomValues=a=>a instanceof Uint32Array && a.length===1 ? a.fill(42) : random(a);
      window.identityLeaks=[];
      window.revealFirstFrames={};
      new MutationObserver(()=>{
        if(/Checkpoint|v016-belief|iteration/i.test(document.body?.innerText||'')) window.identityLeaks.push(document.body.innerText);
        const marker=document.querySelector('.exposure-marker'),piece=document.querySelector('.evader-ring');
        if(marker && piece && !window.revealFirstFrames[marker.dataset.round])
          window.revealFirstFrames[marker.dataset.round]={destination:Number(marker.dataset.destination),x:Number(piece.getAttribute('cx')),y:Number(piece.getAttribute('cy')),radius:Number(marker.querySelector('circle').getAttribute('r')),stroke:getComputedStyle(marker.querySelector('circle')).stroke,hasText:!!marker.querySelector('text,rect,title')};
      }).observe(document,{subtree:true,childList:true,characterData:true});
    ''')
    requests=[]
    fail_preload=True
    def weights_route(route):
        requests.append(route.request.url)
        if fail_preload and catalog['pursuer'][2]['url'].removeprefix('./') in route.request.url:
            route.abort()
        else:
            route.continue_()
    context.route('**/scotlandyard/models/*.onnx',weights_route)
    page=context.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url);ready(page)
    assert page.locator('#evaluation-start').inner_text()=='开始测评'
    assert data(page) is None
    page.screenshot(path='/tmp/scotland-blind-start.png',full_page=True)
    assert requests==[]
    page.locator('#evaluation-start').click()
    page.locator('#error:not([hidden])').wait_for(timeout=120000)
    page.wait_for_function("document.querySelector('#app').getAttribute('aria-busy')==='false'")
    failed=data(page)
    assert failed['active'] is None
    assert all(g['result']=='not_started' for g in failed['batches'][0]['games'])
    assert len(requests)==3
    fail_preload=False
    page.locator('#evaluation-start').click();ready(page)
    assert len(requests)==len(catalog['pursuer'])+1
    assert len(set(requests))==len(catalog['pursuer'])
    assert page.locator('#model-loading').is_hidden()
    assert page.locator('#model-progress').get_attribute('value')=='100'
    saved=data(page);batch=saved['batches'][0]
    assert batch['batch_id']==failed['batches'][0]['batch_id']
    assert [(g['game_id'],g['model_id'],g['seed']) for g in batch['games']]==[(g['game_id'],g['model_id'],g['seed']) for g in failed['batches'][0]['games']]
    print('PASS all weights downloaded before play, failed preload preserves schedule, retry reuses verified weights',flush=True)
    context.set_offline(True)
    offline_requests=len(requests)
    counts={cp:sum(saved['models'][g['model_id']]['checkpoint']==cp for g in batch['games']) for cp in [2,8,16,32,118]}
    assert counts=={2:2,8:5,16:7,32:4,118:2}
    assert saved['models'][saved['active']['model_id']]['checkpoint']==2
    public=page.evaluate("async()=>{const {requestJson}=await import('./local-api.js?v=v016-preload-1');return requestJson('/api/catalog')}")
    assert 'pursuer' not in public
    assert page.locator('#quick-pursuer').count()==0
    reveals=[]
    for _ in range(22):
        if state(page)['terminal']:break
        before=state(page);round_number=before['round'];move=greedy(page)
        if round_number==3:page.emulate_media(reduced_motion='no-preference')
        if round_number in [3,8,13,18]:
            assert '本次移动后' in page.locator('#reveal-hint').inner_text()
            assert page.locator(f'.round-cell[data-round="{round_number}"] small').inner_text()=='待揭示'
        play(page,move)
        if round_number in [3,8,13,18]:
            frame=page.evaluate('(r)=>window.revealFirstFrames[r]',round_number)
            assert frame['destination']==move['destination']
            assert frame['radius']>float(page.locator('.evader-ring').get_attribute('r'))
            assert frame['stroke']!='none'
            assert frame['hasText'] is False
            assert page.locator('.exposure-marker').count()==0
            assert page.locator('#last-reveal,.exposure-label,.exposure-caption').count()==0
            assert page.locator(f'.round-cell[data-round="{round_number}"] small').inner_text()=='已揭示'
            reveals.append(round_number)
        if round_number==3:
            first_frame=page.evaluate('window.revealFirstFrames[3]')
            destination=next(n for n in board['nodes'] if n['id']==move['destination'])
            assert {k:first_frame[k] for k in ['destination','x','y']}=={'destination':move['destination'],'x':destination['x'],'y':destination['y']}
            assert page.locator('#board > g').last.get_attribute('class')=='exposure-layer'
            page.screenshot(path='/tmp/scotland-blind-reveal.png',full_page=True)
            page.emulate_media(reduced_motion='reduce')
            page.locator('#undo').click();ready(page)
            assert page.locator('.exposure-marker').count()==0
            assert '本次移动后' in page.locator('#reveal-hint').inner_text()
            assert page.locator('.round-cell[data-round="3"] small').inner_text()=='待揭示'
            play(page,move)
            assert page.locator('.exposure-marker').count()==0
            partial=export(page,'scotland-blind-partial.json')
            assert partial['batches'][0]['num_games']==0
            assert len(partial['batches'][0]['games'])==1
            assert partial['batches'][0]['games'][0]['result']=='in_progress'
            assert partial['batches'][0]['games'][0]['undo_num']==1
        if round_number==8:
            before_close=data(page)
            assert not page.evaluate('window.identityLeaks')
            assert len(requests)==offline_requests
            context.set_offline(False)
            page.close();page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(url);ready(page)
            assert data(page)==before_close
            assert len(requests)==offline_requests+len(catalog['pursuer'])
            offline_requests=len(requests)
            context.set_offline(True)
    assert reveals==[3,8,13,18]
    assert data(page)['active']['game']['result']=='win'
    assert page.locator('#undo').is_disabled()
    assert page.locator('#evaluation-next').is_visible()
    if args.reveals_only:
        assert not errors,errors
        print('PASS reveal-only rings at all four rounds, hidden next round, no captions, undo, refresh and normal animation',flush=True)
        browser.close()
        raise SystemExit(0)
    print('PASS hidden identity, fixed quotas, all four reveal boundaries, undo, close/reopen and partial export',flush=True)
    captured=False
    for number in range(2,21):
        page.locator('#evaluation-next').click();ready(page)
        assert f'第 {number}/20 局' in page.locator('#evaluation-heading').inner_text()
        model=data(page)['models'][data(page)['active']['model_id']]['checkpoint']
        if model==118 and not captured:
            for _ in range(22):
                if state(page)['terminal']:break
                page.locator('.move-btn').first.click();ready(page)
            assert state(page)['winner']=='pursuers'
            assert data(page)['active']['game']['result']=='in_progress'
            last_move=state(page)['last_evader_move']
            for count in [1,2]:
                page.locator('#undo').click();ready(page)
                assert data(page)['active']['game']['undo_num']==count
                play(page,last_move)
                assert state(page)['winner']=='pursuers'
            captured=True
        page.locator('#give-up').click();ready(page)
        assert len([g for g in data(page)['batches'][0]['games'] if g['result'] in ['win','give_up']])==number
    assert captured
    assert page.locator('#evaluation-next').is_hidden()
    assert page.locator('#evaluation-repeat').is_visible()
    assert '20/20' in page.locator('#evaluation-progress-text').inner_text()
    completed=export(page,'scotland-blind-completed.json')
    assert completed['batches'][0]['status']=='completed'
    assert sum(r['num_games'] for r in completed['summary'])==20
    assert {r['checkpoint']:r['num_games'] for r in completed['summary']}==counts
    assert all(set(g)=={'game_id','game_number','model_id','checkpoint','result','game_len','undo_num'} for g in completed['batches'][0]['games'])
    for row in completed['summary']:
        games=[g for g in completed['batches'][0]['games'] if g['model_id']==row['model_id']]
        assert row['avg_undo_num']==sum(g['undo_num'] for g in games)/len(games)
        assert row['avg_gamelen']==sum(g['game_len'] for g in games)/len(games)
    assert len(requests)==offline_requests
    page.screenshot(path='/tmp/scotland-blind-complete.png',full_page=True)
    page.locator('#evaluation-repeat').click();ready(page)
    assert len(requests)==offline_requests
    context.set_offline(False)
    before_reload=data(page)
    page.reload();ready(page)
    assert data(page)==before_reload
    assert data(page)['batches'][0]['completed_at']==completed['batches'][0]['completed_at']
    assert len(data(page)['batches'])==2
    assert data(page)['participant_id']==completed['participant_id']
    assert data(page)['batches'][1]['batch_id']!=completed['batches'][0]['batch_id']
    cumulative=export(page,'scotland-blind-repeat.json')
    assert cumulative['batches'][0]==completed['batches'][0]
    assert cumulative['batches'][1]['games'][0]['result']=='in_progress'
    assert page.evaluate("localStorage.getItem('skz_scotland_v016_unlimited_undo_v1')")=='old pilot results'
    for width,height in [(1280,720),(390,844)]:
        page.set_viewport_size({'width':width,'height':height})
        page.locator('#evaluation-export').scroll_into_view_if_needed()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='/tmp/scotland-blind-mobile.png',full_page=True)
    assert not page.evaluate('window.identityLeaks')
    assert not errors,errors
    print('PASS offline 20-game batch and repeat without downloads, capture retries, hidden identity, exact totals, refresh, cumulative exports and responsive layout',flush=True)
    for model in catalog['pursuer']:
        fixtures=json.loads((root/f"tests/{model['id']}-fixtures.json").read_text())
        result=page.evaluate('''async ({model,fixtures})=>{
          const worker=new Worker(new URL('./inference-worker.js',location.href));
          let id=0;
          const call=message=>new Promise((resolve,reject)=>{
            const timer=setTimeout(()=>reject(new Error('fixture timeout')),60000);
            worker.onmessage=({data})=>{if(data.progress)return;clearTimeout(timer);data.error?reject(new Error(data.error)):resolve(data.result);};
            worker.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};
            worker.postMessage({id:++id,...message});
          });
          try {
            await call({kind:'prepare',models:[{model,url:new URL(model.url,location.href).href}]});
            await call({kind:'load',model});
            const outputs=[];
            for(const f of fixtures) outputs.push(await call({kind:'run',inputs:Object.fromEntries(Object.entries(f.inputs).map(([k,v])=>[k,v.data]))}));
            return outputs;
          } finally {worker.terminate();}
        }''',{'model':model,'fixtures':fixtures})
        for actual,fixture in zip(result,fixtures):
            assert actual['actions']==fixture['actions'];assert actual['order']==fixture['order']
        print('WASM PARITY',model['id'],len(fixtures),flush=True)
    browser.close()
