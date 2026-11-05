(function (scope) {
    const app=document.querySelector('.board-page .app');
    const main=app?.querySelector('.main');
    const card=main?.querySelector('.board-card');
    const column=main?.querySelector('.board-col');
    const board=document.getElementById('board');
    if(!main || !card || !board)return;
    let last='';
    const dimensions=element=>{
        const css=getComputedStyle(element);
        return {x:parseFloat(css.paddingLeft)+parseFloat(css.paddingRight)+parseFloat(css.borderLeftWidth)+parseFloat(css.borderRightWidth),
            y:parseFloat(css.paddingTop)+parseFloat(css.paddingBottom)+parseFloat(css.borderTopWidth)+parseFloat(css.borderBottomWidth)};
    };
    const update=()=>{
        const inset=dimensions(card),appCSS=getComputedStyle(app),gridCSS=getComputedStyle(main);
        const gap=parseFloat(gridCSS.columnGap)||0;
        const width=main.clientWidth;
        let extra=0;
        if(column!==card)for(const child of column.children)if(child!==card && child.getClientRects().length)extra+=child.getBoundingClientRect().height+(parseFloat(getComputedStyle(column).rowGap)||0);
        const height=Math.max(1,innerHeight-(main.getBoundingClientRect().top+scrollY)-parseFloat(appCSS.paddingBottom)-extra);
        const box=board.viewBox?.baseVal;
        const ratio=box?.height?box.width/box.height:1;
        const wrap=board.closest('.board-wrap');
        const wrapInset=wrap?dimensions(wrap):{x:0,y:0};
        const desired=Math.max(1,(height-inset.y-wrapInset.y)*ratio+inset.x+wrapInset.x);
        const phone=innerWidth<=720;
        const stacked=phone || desired+400+gap*2>width;
        const boardWidth=Math.min(width,desired);
        const room=Math.max(0,width-boardWidth-gap*2);
        const left=Math.min(320,Math.max(180,Math.min(room-220,room*0.46)));
        const right=Math.min(340,Math.max(220,room-left));
        const columns=stacked?(phone?'minmax(0, 1fr)':'repeat(2, minmax(0, 1fr))'):`${left}px minmax(0, 1fr) ${right}px`;
        const signature=[columns,boardWidth,height,stacked].join('|');
        if(signature===last)return;
        last=signature;
        document.body.dataset.boardLayout=stacked?'stacked':'columns';
        main.style.setProperty('--board-columns',columns);
        card.style.width=`${boardWidth}px`;
        document.documentElement.style.setProperty('--board-height',`${height}px`);
        scope.SkyZeroBoardLayout.stacked=stacked;
        document.dispatchEvent(new Event('boardlayoutchange'));
    };
    scope.SkyZeroBoardLayout={stacked:false,update,
        fitCanvas({size,margin}) {
            update();
            const inset=dimensions(card);
            const logical=Math.max(1,card.getBoundingClientRect().width-inset.x);
            margin=Math.min(margin,logical/4);
            return {logical,margin,cell:(logical-2*margin)/(size-1)};
        }
    };
    const observer=new ResizeObserver(update);
    observer.observe(app);observer.observe(main);observer.observe(app.querySelector('.topbar'));
    new MutationObserver(update).observe(board,{attributes:true,attributeFilter:['viewBox']});
    window.addEventListener('resize',update);
    update();
})(window);
