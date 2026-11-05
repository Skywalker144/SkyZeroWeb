const TEXT = {
    zh: {
        title: "六角棋", auto: "自动主题", light: "明亮", dark: "暗黑",
        evaluation: "胜率", gameStats: "本局统计", totalMoves: "总手数", searchVisits: "AI 搜索",
        candidates: "候选着法", candidateLegend: "胜率 · 访问", noAnalysis: "暂无分析", model: "模型",
        size: "棋盘大小", side: "执子", budget: "思考", blackFirst: "黑棋",
        whiteSecond: "白棋", balancedOpening: "平衡开局", cancelOpening: "取消生成", balancing: "生成平衡开局中…",
        attempts: "次尝试", whiteWin: "白胜", newGame: "新对局", undo: "悔棋",
        downloading: "下载模型…", loading: "初始化 AI…", yourTurn: "轮到你", thinking: "AI 思考中…",
        won: "你赢了！", lost: "AI 获胜", error: "加载或推理失败", retry: "点击新对局重试",
        last: "最新轮次", hours: "小时", opening: "开局"
    },
    en: {
        title: "", auto: "Auto theme", light: "Light", dark: "Dark",
        evaluation: "Win rate", gameStats: "Game statistics", totalMoves: "Moves", searchVisits: "AI visits",
        candidates: "Candidate moves", candidateLegend: "Win % · visits", noAnalysis: "No analysis yet", model: "Model",
        size: "Board size", side: "Side", budget: "Think", blackFirst: "Black",
        whiteSecond: "White", balancedOpening: "Balanced opening", cancelOpening: "Cancel generation", balancing: "Generating opening…",
        attempts: "attempts", whiteWin: "White", newGame: "New game", undo: "Undo",
        downloading: "Downloading model…", loading: "Initializing AI…", yourTurn: "Your turn", thinking: "AI is thinking…",
        won: "You won!", lost: "AI won", error: "Loading or inference failed", retry: "Click New game to retry",
        last: "Latest iteration", hours: "hours", opening: "Opening"
    }
};
const $ = id => document.getElementById(id);
let language = localStorage.getItem('skz_lang') === 'en' ? 'en' : 'zh';
let manifest, worker, ready = false, thinking = false, balancing = false, requestId = 0, activeId = 0;
let opening = null, openingAttempts = 0;
let game = new HexEngine.HexGame(15), board = new Int8Array(225), history = [], error = '', phase = 'downloading', stats = null;
const player = () => history.length % 2 ? -1 : 1;
const human = () => Number($('side').value);
const t = key => TEXT[language][key];
const undoFloor = () => opening ? (human() === 1 ? 2 : 1) : 0;

function renderText() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(element => { element.textContent = t(element.dataset.i18n); });
    $('language').textContent = language === 'zh' ? 'EN' : '中';
    if (manifest) {
        for (const model of manifest.models) {
            $('model').querySelector(`option[value="${model.id}"]`).textContent = model.id === 'last' ? t('last') : `${model.target_seconds / 3600} ${t('hours')}`;
        }

    }
    const winner = game.winner(board);
    $('status').title = error;
    $('status').textContent = error ? t('error') : !ready ? t(phase)
        : balancing ? `${t('balancing')} ${openingAttempts} ${t('attempts')}` : winner ? t(winner === human() ? 'won' : 'lost') : thinking ? t('thinking') : t('yourTurn');
    BoardGameAnalysis.loading(!ready && !error, $('status').textContent);
    $('status-pill').dataset.variant = error ? 'error' : !ready || balancing || thinking ? 'thinking' : winner ? 'done' : 'active';
    BoardGameAnalysis.render({ply: history.length, result: stats, t, coordinate: action => `${String.fromCharCode(65 + action % game.size)}${Math.floor(action / game.size) + 1}`});
    $('undo').disabled = !ready || balancing || history.length <= undoFloor();
    $('balanced-opening').disabled = !ready || !!error;
    $('balanced-opening').textContent = t(balancing ? 'cancelOpening' : 'balancedOpening');
    $('opening-info').textContent = opening ? `${opening.attempts} ${t('attempts')} · ${t('whiteWin')} ${Math.round(opening.whiteWinProbability * 100)}%` : '';
    $('opening-info').hidden = !opening;
    $('board').setAttribute('aria-disabled', String(!ready || thinking || balancing || !!winner || !!error));
}

function renderBoard() {
    const size = game.size, radius = 13, dx = Math.sqrt(3) * radius;
    const center = (row, col) => [34 + dx * (col + row / 2), 36 + radius * 1.5 * row];
    const vertex = (x, y, i) => [x + radius * Math.cos((60 * i - 30) * Math.PI / 180), y + radius * Math.sin((60 * i - 30) * Math.PI / 180)];
    const winner = game.winner(board), path = new Set(winner ? game.connection(board, winner) : []);
    const [right, bottom] = center(size - 1, size - 1);
    $('board').setAttribute('viewBox', `0 0 ${right + 34} ${bottom + 34}`);
    const markup = ['<defs><radialGradient id="hex-black" cx="30%" cy="30%"><stop stop-color="var(--stone-black-0)"/><stop offset="1" stop-color="var(--stone-black-1)"/></radialGradient><radialGradient id="hex-white" cx="30%" cy="30%"><stop stop-color="var(--stone-white-0)"/><stop offset="1" stop-color="var(--stone-white-1)"/></radialGradient></defs>'];
    for (let row = 0; row < size; row++) for (let col = 0; col < size; col++) {
        const action = row * size + col, [x, y] = center(row, col);
        const color = board[action] === 1 ? 'black' : board[action] === -1 ? 'white' : 'empty';
        const coordinate = `${String.fromCharCode(65 + col)}${row + 1}`;
        markup.push(`<polygon class="cell ${color}${path.has(action) ? ' connected' : ''}" data-action="${action}" points="${Array.from({length: 6}, (_, i) => vertex(x, y, i).join(',')).join(' ')}" role="button" tabindex="${board[action] ? -1 : 0}" aria-label="${coordinate}" aria-disabled="${!!board[action]}"><title>${coordinate}</title></polygon>`);
        if (board[action]) markup.push(`<circle class="hex-piece ${color}" cx="${x}" cy="${y}" r="${radius * .78}"/>`);
        if (history.at(-1) === action) markup.push(`<circle class="last-marker" cx="${x}" cy="${y}" r="4" stroke="${board[action] === 1 ? '#fff' : '#24313d'}"/>`);
        if (row === 0) markup.push(`<text class="board-label" x="${x}" y="${y - 21}">${String.fromCharCode(65 + col)}</text>`);
        if (col === 0) markup.push(`<text class="board-label" x="${x - 24}" y="${y + 3}">${row + 1}</text>`);
    }
    for (const [edge, color] of [['top', '#24313d'], ['bottom', '#24313d'], ['left', '#fff'], ['right', '#fff']]) {
        for (let i = 0; i < size; i++) {
            const [x, y] = center(edge === 'top' ? 0 : edge === 'bottom' ? size - 1 : i,
                edge === 'left' ? 0 : edge === 'right' ? size - 1 : i);
            const vertices = edge === 'top' ? [4, 5, 0] : edge === 'bottom' ? [1, 2, 3] : edge === 'left' ? [3, 4] : [0, 1];
            const points = vertices.map(index => vertex(x, y, index).join(',')).join(' ');
            markup.push(`<polyline points="${points}" fill="none" stroke="#66717d" stroke-width="6" pointer-events="none"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="4" pointer-events="none"/>`);
        }
    }
    $('board').innerHTML = markup.join('');
    renderText();
}

function startAI() {
    if (!ready || error || balancing || game.winner(board) || player() === human()) { renderText(); return; }
    thinking = true;
    stats = null;
    activeId = ++requestId;
    renderText();
    worker.postMessage({id: activeId, type: 'search', board: Array.from(board), player: player(), size: game.size,
        maxVisits: 512, thinkMs: Number($('budget').value)});
}

function reset(start = true) {
    activeId = ++requestId;
    worker?.postMessage({type: 'cancel'});
    game = new HexEngine.HexGame(Number($('size').value) || 15);
    board = new Int8Array(game.size ** 2);
    history = [];
    BoardGameAnalysis.reset();
    thinking = false;
    balancing = false;
    opening = null;
    openingAttempts = 0;
    stats = null;
    renderBoard();
    if (start) startAI();
}

function loadModel() {
    worker?.terminate();
    ready = false;
    error = '';
    phase = 'downloading';
    reset();
    worker = new Worker('/hex/worker.js?v=20261105b');
    const id = activeId = ++requestId;
    worker.onerror = event => { error = event.message; thinking = false; balancing = false; renderText(); };
    worker.onmessage = ({data}) => {
        if (data.id !== activeId) return;
        if (data.type === 'loading') phase = 'loading';
        if (data.type === 'ready') { ready = true; startAI(); }
        if (data.type === 'error') { error = data.message; thinking = false; balancing = false; }
        if (data.type === 'opening-progress') openingAttempts = data.result.attempts;
        if (data.type === 'opening') {
            balancing = false;
            opening = data.result;
            board = game.play(board, opening.action, 1);
            history.push(opening.action);
            renderBoard();
            startAI();
        }
        if (data.type === 'progress') stats = data.result;
        if (data.type === 'move') {
            thinking = false;
            stats = data.result;
            BoardGameAnalysis.record(history.length + 1, stats);
            board = game.play(board, data.result.action, player());
            history.push(data.result.action);
            renderBoard();
        }
        renderText();
    };
    const model = manifest.models.find(item => item.id === $('model').value);
    worker.postMessage({id, type: 'init', canvas: manifest.network.canvas, url: `/hex/models/${model.file}`});
}

function chooseCell(event) {
    const cell = event.target.closest('[data-action]');
    if (!cell || !ready || error || thinking || balancing || player() !== human() || game.winner(board)) return;
    const action = Number(cell.dataset.action);
    if (board[action]) return;
    board = game.play(board, action, player());
    history.push(action);
    renderBoard();
    startAI();
}

$('board').addEventListener('click', chooseCell);
$('board').addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); chooseCell(event); }
});
$('model').addEventListener('change', loadModel);
for (const id of ['size', 'side']) $(id).addEventListener('change', () => ready ? reset() : manifest && loadModel());
$('new-game').addEventListener('click', () => {
    if (error) { if (manifest) loadModel(); else loadManifest(); }
    else if (ready) reset();
});
$('balanced-opening').addEventListener('click', () => {
    if (balancing) { reset(); return; }
    if (!ready || error) return;
    reset(false);
    balancing = true;
    activeId = ++requestId;
    renderText();
    worker.postMessage({id: activeId, type: 'opening', size: game.size, config: manifest.hex_opening});
});
$('undo').addEventListener('click', () => {
    activeId = ++requestId;
    worker.postMessage({type: 'cancel'});
    thinking = false;
    do { history.pop(); } while (history.length > undoFloor() && player() !== human());
    board = new Int8Array(game.size ** 2);
    for (let i = 0; i < history.length; i++) board = game.play(board, history[i], i % 2 ? -1 : 1);
    stats = null;
    BoardGameAnalysis.trim(history.length);
    renderBoard();
    startAI();
});
$('language').addEventListener('click', () => {
    language = language === 'zh' ? 'en' : 'zh';
    localStorage.setItem('skz_lang', language);
    renderText();
});
$('theme').value = localStorage.getItem('skz_theme') || 'auto';
function applyTheme() {
    const mode = $('theme').value;
    document.documentElement.dataset.theme = mode === 'auto' ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : mode;
}
$('theme').addEventListener('change', () => { localStorage.setItem('skz_theme', $('theme').value); applyTheme(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
async function loadManifest() {
    error = '';
    renderText();
    try {
        const response = await fetch('/hex/models/manifest.json?v=20261009');
        if (!response.ok) throw new Error(`Manifest HTTP ${response.status}`);
        manifest = await response.json();
        $('model').replaceChildren();
        $('size').replaceChildren();
        for (const model of manifest.models) $('model').add(new Option(model.id, model.id));
        $('model').value = manifest.default;
        $('model').disabled = false;
        for (const size of manifest.sizes) $('size').add(new Option(`${size} × ${size}`, size));
        $('size').value = manifest.network.canvas;
        $('size').disabled = false;
        loadModel();
    } catch (problem) { error = problem.message; renderText(); }
}
renderBoard();
loadManifest();
