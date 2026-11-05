(function () {
    const labels = {
        zh: {settings:'设置',immediate:'更改会立即应用',gameBoard:'对局与棋盘',theme:'主题',boardSize:'棋盘大小',close:'关闭设置'},
        en: {settings:'Settings',immediate:'Changes apply immediately',gameBoard:'Game and board',theme:'Theme',boardSize:'Board size',close:'Close settings'}
    };
    const translate = () => {
        const text = labels[document.documentElement.lang.startsWith('zh') ? 'zh' : 'en'];
        document.querySelectorAll('[data-ui]').forEach(node => {node.textContent=text[node.dataset.ui];});
        document.getElementById('settings_close_btn')?.setAttribute('aria-label',text.close);
    };
    new MutationObserver(translate).observe(document.documentElement,{attributes:true,attributeFilter:['lang']});
    translate();
    const button=document.getElementById('settings_btn'),modal=document.getElementById('settings_pop'),close=document.getElementById('settings_close_btn');
    if (button && modal && close) {
        const setOpen = open => {
            modal.classList.toggle('hidden',!open);
            document.body.classList.toggle('settings-open',open);
            button.setAttribute('aria-expanded',String(open));
            (open ? close : button).focus();
        };
        button.addEventListener('click',()=>setOpen(modal.classList.contains('hidden')));
        close.addEventListener('click',()=>setOpen(false));
        modal.addEventListener('click',event=>{if(event.target===modal)setOpen(false);});
        document.addEventListener('keydown',event=> {
            if(modal.classList.contains('hidden'))return;
            if(event.key==='Escape')setOpen(false);
            if(event.key==='Tab') {
                const nodes=Array.from(modal.querySelectorAll('button,select,input,[tabindex="0"]')).filter(node=>!node.disabled && !node.classList.contains('control-source') && node.getClientRects().length);
                const first=nodes[0],last=nodes.at(-1);
                if(event.shiftKey && document.activeElement===first){event.preventDefault();last.focus();}
                else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first.focus();}
            }
        });
        const wrap=button.closest('.settings-wrap'),top=document.querySelector('.topbar-right'),bottom=document.getElementById('mobile_actionbar');
        if(wrap && top && bottom) {
            const query=matchMedia('(max-width:720px)');
            const place=()=>{(query.matches ? bottom : top).appendChild(wrap);};
            query.addEventListener('change',place);place();
        }
    }
    for(const select of document.querySelectorAll('select[data-board-segment],select[data-board-select]')) {
        const segmented=select.hasAttribute('data-board-segment');
        const container=document.createElement('div');
        container.className=segmented ? 'seg-row seg-slide' : 'custom-select';
        select.after(container);
        select.classList.add('control-source');
        select.tabIndex=-1;
        let trigger,menu;
        if(!segmented) {
            trigger=document.createElement('button');trigger.type='button';trigger.className='cs-trigger';
            trigger.setAttribute('aria-haspopup','listbox');trigger.setAttribute('aria-expanded','false');
            menu=document.createElement('div');menu.className='cs-menu hidden';menu.setAttribute('role','listbox');
            container.append(trigger,menu);
            trigger.addEventListener('click',()=> {
                const open=menu.classList.toggle('hidden')===false;
                trigger.setAttribute('aria-expanded',String(open));
                if(open)(menu.querySelector('[aria-selected="true"]') || menu.querySelector('button'))?.focus();
            });
            document.addEventListener('click',event=>{if(!container.contains(event.target)){menu.classList.add('hidden');trigger.setAttribute('aria-expanded','false');}});
            container.addEventListener('keydown',event=> {
                if(event.key==='Escape'){menu.classList.add('hidden');trigger.setAttribute('aria-expanded','false');trigger.focus();}
                if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)) {
                    event.preventDefault();menu.classList.remove('hidden');trigger.setAttribute('aria-expanded','true');
                    const nodes=Array.from(menu.querySelectorAll('button:not(:disabled)'));
                    const index=nodes.indexOf(document.activeElement);
                    nodes[event.key==='Home' ? 0 : event.key==='End' ? nodes.length-1 : (index+(event.key==='ArrowDown'?1:-1)+nodes.length)%nodes.length]?.focus();
                }
            });
        }
        const render=()=> {
            const target=segmented ? container : menu;
            const buttons=new Map(Array.from(target.querySelectorAll('button')).map(node=>[node.dataset.value,node]));
            for(const option of select.options) {
                let node=buttons.get(option.value);
                if(!node) {
                    node=document.createElement('button');node.type='button';node.dataset.value=option.value;
                    node.className=segmented?'seg-btn':'cs-option';
                    if(!segmented)node.setAttribute('role','option');
                    node.addEventListener('click',()=> {
                        select.value=node.dataset.value;select.dispatchEvent(new Event('change',{bubbles:true}));
                        if(menu){menu.classList.add('hidden');trigger.setAttribute('aria-expanded','false');trigger.focus();}
                        render();
                    });
                    target.append(node);
                }
                if(node.textContent!==option.text)node.textContent=option.text;
                node.disabled=select.disabled || option.disabled;
                node.setAttribute(segmented?'aria-pressed':'aria-selected',String(option.selected));
                buttons.delete(option.value);
            }
            for(const node of buttons.values())node.remove();
            const name=document.querySelector(`label[for="${select.id}"]`)?.textContent || select.id;
            if(trigger) {trigger.textContent=(select.selectedOptions[0]?.text || '—')+' ▾';trigger.disabled=select.disabled;trigger.setAttribute('aria-label',name);}
            else container.setAttribute('aria-label',name);
        };
        select.addEventListener('change',render);
        new MutationObserver(render).observe(select,{childList:true,subtree:true,characterData:true,attributes:true});
        render();
    }
})();
