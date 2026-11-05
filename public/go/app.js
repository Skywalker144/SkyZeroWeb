const $=id=>document.getElementById(id);
const TEXT = {
    zh: {
        title: "围棋", auto: "自动主题", light: "明亮", dark: "暗黑",
        side: "执子", blackFirst: "黑棋", whiteSecond: "白棋", budget: "思考",
        newGame: "新对局", undo: "悔棋", pass: "停一手", gameStats: "本局统计",
        totalMoves: "总手数", searchVisits: "AI 搜索", candidates: "候选着法", candidateLegend: "胜率 · 访问",
        noAnalysis: "暂无分析", evaluation: "胜率", source: "规则源码", retry: "重试",
        loading: "加载模型…", preparing: "准备棋局…", thinking: "AI 思考中…", yourTurn: "轮到你",
        win: "你赢了！", loss: "AI 获胜", draw: "和棋", black: "黑方",
        white: "白方", error: "加载或运行失败，请重试。", ruleLabel: "规则", ruleSummary: "中国规则 · 贴 7.5 目"
    },
    en: {
        title: "Go", auto: "Auto theme", light: "Light", dark: "Dark",
        side: "Side", blackFirst: "Black", whiteSecond: "White", budget: "Think",
        newGame: "New game", undo: "Undo", pass: "Pass", gameStats: "Game statistics",
        totalMoves: "Moves", searchVisits: "AI visits", candidates: "Candidate moves", candidateLegend: "Win % · visits",
        noAnalysis: "No analysis yet", evaluation: "Win rate", source: "Rules source", retry: "Retry",
        loading: "Loading model…", preparing: "Preparing position…", thinking: "AI is thinking…", yourTurn: "Your turn",
        win: "You win!", loss: "AI wins", draw: "Draw", black: "Black",
        white: "White", error: "Loading or execution failed. Please retry.", ruleLabel: "Rules", ruleSummary: "Chinese · 7.5 komi"
    }
};
let lang=localStorage.getItem('skz_lang')==='en'?'en':'zh';
const t=key=>TEXT[lang][key] || key;
let human=1;
let session;
const size=()=>Number($('size').value);
const coordinate=action=>action===size()**2?t('pass'):`${'ABCDEFGHJKLMNOPQRST'[action%size()]}${size()-Math.floor(action/size())}`;
function renderBoard() {
    const n=size(),step=32,pad=36,extent=pad*2+step*(n-1),legal=new Set(session.state?.legal || []);
    const disabled=!session.ready || session.busy || session.thinking || !!session.error || !session.state || session.state.finished || session.state.player!==human;
    const focused=document.activeElement?.dataset.action;
    $('board').setAttribute('viewBox',`0 0 ${extent} ${extent}`);
    let html='<defs><radialGradient id="go-black" cx="30%" cy="30%"><stop stop-color="#555"/><stop offset="1" stop-color="#000"/></radialGradient><radialGradient id="go-white" cx="30%" cy="30%"><stop stop-color="#fff"/><stop offset="1" stop-color="#d5d5d5"/></radialGradient></defs><g class="go-grid">';
    for(let i=0;i<n;i++){const p=pad+i*step;html+=`<path d="M${pad} ${p}H${extent-pad}M${p} ${pad}V${extent-pad}"/>`;}
    html+='</g>';
    const stars=n===19?[3,9,15]:n===13?[3,6,9]:[2,4,6];
    for(const y of stars)for(const x of stars)if(n!==9 || x===4 && y===4 || x!==4 && y!==4)html+=`<circle class="go-star" cx="${pad+x*step}" cy="${pad+y*step}" r="2.8"/>`;
    for(let i=0;i<n;i++) {
        const p=pad+i*step;
        html+=`<text class="go-label" x="${p}" y="15">${'ABCDEFGHJKLMNOPQRST'[i]}</text><text class="go-label" x="${p}" y="${extent-15}">${'ABCDEFGHJKLMNOPQRST'[i]}</text><text class="go-label" x="15" y="${p}">${n-i}</text><text class="go-label" x="${extent-15}" y="${p}">${n-i}</text>`;
    }
    for(let a=0;a<n*n;a++) {
        const x=pad+a%n*step,y=pad+Math.floor(a/n)*step,piece=session.state?.board[a] || 0,enabled=!disabled && legal.has(a);
        html+=`<g class="go-point" data-action="${a}" role="button" tabindex="${enabled?0:-1}" aria-label="${coordinate(a)}${piece?' '+t(piece===1?'black':'white'):''}" aria-disabled="${!enabled}"><rect class="go-hit" x="${x-16}" y="${y-16}" width="32" height="32"/>`;
        if(piece)html+=`<circle class="go-stone go-${piece===1?'black':'white'}" cx="${x}" cy="${y}" r="14.7"/>`;
        if(session.actions.at(-1)===a)html+=`<circle class="go-last" cx="${x}" cy="${y}" r="4" stroke="${piece===1?'#fff':'#111'}"/>`;
        html+='</g>';
    }
    $('board').innerHTML=html;
    $('board').setAttribute('aria-disabled',String(disabled));
    if(focused!==undefined)$('board').querySelector(`[data-action="${focused}"]`)?.focus();
}
function render() {
    document.documentElement.lang=lang==='zh'?'zh-CN':'en';
    document.title=`${t('title')} · SkyZero`;
    document.querySelectorAll('[data-i18n]').forEach(node=>{node.textContent=t(node.dataset.i18n);});
    $('language').textContent=lang==='zh'?'EN':'中';
    let status=!session.ready?t('loading'):session.busy?t('preparing'):session.thinking?t('thinking'):t('yourTurn');
    if(session.state?.finished)status=t(session.state.winner===0?'draw':session.state.winner===human?'win':'loss');
    if(session.state?.finished && session.state.winner)status+=` · ${t(session.state.score>0?'white':'black')} +${Math.abs(session.state.score)}`;
    $('status').title=session.error;
    $('status').textContent=session.error?t('error'):status;
    $('status-pill').dataset.variant=session.error?'error':!session.ready || session.busy || session.thinking?'thinking':session.state?.finished?'done':'active';
    $('undo').disabled=!session.ready || session.actions.length<=(human===-1?1:0);
    $('pass').disabled=!session.ready || session.busy || session.thinking || !!session.error || !session.state || session.state.finished || session.state.player!==human;
    $('retry').hidden=!session.error;
    BoardGameAnalysis.render({ply:session.actions.length,result:session.lastSearch,t,coordinate});
    renderBoard();
}
function choose(action) {
    if(!session.ready || session.busy || session.thinking || session.error || !session.state || session.state.finished || session.state.player!==human || !session.state.legal.includes(action))return;
    session.actions.push(action);session.syncPosition();
}
session=new BoardGameSession({game:'go',version:'20261105b',getHuman:()=>human,render,analysis:BoardGameAnalysis,
    positionOptions:()=>({size:size(),komi:session.manifest.game.komi}),
    onManifest:manifest=>{
        $('rules-source').href='/go/'+manifest.rules.source;
        const selected=size();
        $('size').replaceChildren(...manifest.sizes.map(n=>new Option(`${n} × ${n}`,n)));
        $('size').value=manifest.sizes.includes(selected)?selected:manifest.default_size;
    }
});
$('board').addEventListener('click',event=>{const node=event.target.closest('[data-action]');if(node)choose(Number(node.dataset.action));});
$('board').addEventListener('keydown',event=>{if(event.key==='Enter' || event.key===' '){event.preventDefault();const node=event.target.closest('[data-action]');if(node)choose(Number(node.dataset.action));}});
$('pass').onclick=()=>choose(size()**2);
$('new-game').onclick=()=>session.newGame();
$('undo').onclick=()=>session.undo();
$('side').onchange=()=>{human=Number($('side').value);session.newGame();};
$('size').onchange=()=>session.newGame();
$('budget').onchange=()=>{if(session.thinking)session.syncPosition();};
$('retry').onclick=()=>session.boot();
$('language').onclick=()=>{lang=lang==='zh'?'en':'zh';localStorage.setItem('skz_lang',lang);render();};
function applyTheme() {
    const mode=$('theme').value;localStorage.setItem('skz_theme',mode);
    document.documentElement.dataset.theme=mode==='auto'?(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'):mode;
}
$('theme').value=localStorage.getItem('skz_theme') || 'auto';
$('theme').onchange=applyTheme;
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{if($('theme').value==='auto')applyTheme();});
session.boot();
