import os
from playwright.sync_api import sync_playwright
from browser_runtime import use_cached_runtime

BASE=os.environ.get('SKYZERO_WEB_BASE_URL','http://127.0.0.1:8000').rstrip('/')
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page()
    use_cached_runtime(page.context)
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    for game in ['gomoku/lv1','go','xiangqi','hex']:
        for width,height in [(1440,1000),(1920,1080),(1200,900),(1024,900),(390,844),(320,640),(900,480)]:
            page.set_viewport_size(dict(width=width,height=height))
            page.goto(BASE+'/'+game+'/',wait_until='domcontentloaded')
            page.wait_for_function('typeof SkyZeroBoardLayout!=="undefined"')
            page.wait_for_timeout(150)
            result=page.evaluate('''()=>{
                const card=document.querySelector('.board-card').getBoundingClientRect();
                const main=document.querySelector('.main').getBoundingClientRect();
                return {width:card.width,height:card.height,bottom:innerHeight-card.bottom,
                    available:main.width,overflow:document.documentElement.scrollWidth>innerWidth};
            }''')
            assert not result['overflow'],(game,width,height,result)
            assert result['bottom']>=14,(game,width,height,result)
            assert result['bottom']<=20 or result['width']>=result['available']-2,(game,width,height,result)
            if width>=1440 and game!='hex':assert result['height']>=height-130,(game,width,height,result)
    assert not errors,errors
    browser.close()
print('browser-board-layout: four pages fit viewport across desktop, tablet, phone and landscape')
