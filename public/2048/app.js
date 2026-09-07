  (function(){
    'use strict';

    // ---------- i18n (tiny, just for this page) ----------
    const STR = {
      zh: {
        score: '分数', best: '最高分',
        new_game: '新对局', undo: '撤销',
        game_over: '游戏结束', you_win: '到达 2048!',
        play_again: '再来一局', keep_going: '继续',
        brand: 'SkyZero · 2048',
        ai_auto: 'SkyZero 接管', ai_stop: 'SkyZero 接管中',
        ai_loading: '加载SkyZero引擎中…', ai_initializing: '正在初始化SkyZero引擎…',
        ai_ready: 'AI 就绪', ai_error: 'AI 加载失败',
        ai_fast: '更快 更弱', ai_strong: '更慢 更强',
        ai_note: '本局由 SkyZero 接管,未计入排行榜',
      },
      en: {
        score: 'Score', best: 'Best',
        new_game: 'New game', undo: 'Undo',
        game_over: 'Game over', you_win: 'You reached 2048!',
        play_again: 'Play again', keep_going: 'Keep going',
        brand: 'SkyZero · 2048',
        ai_auto: 'SkyZero autopilot', ai_stop: 'SkyZero playing',
        ai_loading: 'Loading SkyZero engine…', ai_initializing: 'Initializing SkyZero engine…',
        ai_ready: 'AI ready', ai_error: 'AI failed to load',
        ai_fast: 'Faster, weaker', ai_strong: 'Slower, stronger',
        ai_note: 'SkyZero played this run — not counted on the leaderboard',
      },
    };
    let LANG = 'zh';
    try {
      const v = localStorage.getItem('skz_lang');
      if (v === 'zh' || v === 'en') LANG = v;
    } catch(_) {}
    function t(k) { return (STR[LANG] && STR[LANG][k]) || STR.en[k] || k; }
    // Render the "SkyZero" in a label as the colored brand wordmark (matches the
    // topbar). Used for the autopilot button + the "not counted" note, which is
    // why those set innerHTML rather than textContent.
    const BRAND_WM = '<span class="brand-wm"><span class="bw-sky">Sky</span><span class="bw-zero">Zero</span></span>';
    const withBrand = (s) => s.replace('SkyZero', BRAND_WM);
    function applyLang() {
      document.documentElement.lang = LANG === 'zh' ? 'zh-CN' : 'en';
      document.getElementById('lbl_score').textContent = t('score');
      document.getElementById('lbl_best').textContent  = t('best');
      document.getElementById('new_btn').textContent   = t('new_game');
      document.getElementById('undo_btn').textContent  = t('undo');
      document.getElementById('ai_speed_fast').textContent = t('ai_fast');
      document.getElementById('ai_speed_strong').textContent = t('ai_strong');
      if (typeof refreshAiSpeed === 'function') refreshAiSpeed();
      if (typeof refreshAiButtons === 'function') refreshAiButtons();
      document.getElementById('overlay_btn').textContent = t('play_again');
      document.title = 'SkyZero · 2048';
      // Refresh overlay title if it's visible (state-dependent label).
      if (typeof updateOverlay === 'function') updateOverlay();
    }

    // ---------- Theme (no in-page switcher; that lives on the home page) ----------
    // Theme + language are chosen on the home page and persisted in localStorage:
    // the inline script in <head> applies the saved theme before first paint and
    // applyLang() applies the saved language at boot. The only thing left to do
    // here is follow OS dark/light changes while the page is open in 'auto' mode.
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (document.documentElement.dataset.themeMode === 'auto') {
          const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          document.documentElement.dataset.theme = dark ? 'dark' : 'light';
        }
      });
    } catch(_) {}

    // ---------- Game ----------
    const SIZE = 4;
    const GAP_PX = 10;
    // Hard ceiling: no tile may ever exceed 131072 (2^17). Two 131072 tiles
    // would merge into 262144, so merging is blocked once a tile hits MAX_TILE.
    const MAX_TILE = 131072;
    const BEST_KEY = 'skz_2048_best';
    const STATE_KEY = 'skz_2048_state';
    const UNDO_LIMIT = 20;
    // Adaptive slide timing. We pick a slide duration per move based on how
    // long it's been since the previous input:
    //   gap ≤ GAP_FAST_MS  → SLIDE_MS_MIN  (rapid swipes; tiles barely linger)
    //   gap ≥ GAP_SLOW_MS  → SLIDE_MS_MAX  (deliberate play; visible motion)
    //   in between         → linear blend
    // The chosen duration is published as `--slide-ms` on `#tiles` and read by
    // every tile's CSS transition.
    const SLIDE_MS_MAX = 120;
    const SLIDE_MS_MIN = 40;
    const GAP_FAST_MS  = 100;
    const GAP_SLOW_MS  = 300;
    const FINALIZE_PAD_MS = 10;   // wait this much after the slide before merging
    let lastInputTs = 0;          // performance.now() of the previous move

    /** @type {number[][]} */
    let grid;
    /** @type {number[][]} parallel to `grid`; 0 = empty, else tile id */
    let tileIds;
    let score = 0;
    let best  = 0;
    let history = [];          // stack of {grid, score, won} snapshots
    let won = false;
    let overlayMode = null;    // null | 'lose' | 'win'
    // Set once a run is no longer leaderboard-eligible: a move was undone.
    // Persisted in saveState so a mid-game reload can't launder it. Reset on
    // newGame().
    let tainted = false;
    // Subset of `tainted`: true specifically when the AI played a move (vs. undo).
    // Drives the "AI was used" note on the end screen. Persisted alongside tainted.
    let aiUsed = false;

    // Live tile DOM, keyed by stable per-tile id. Each entry's DOM node
    // persists across moves so a CSS `transform` transition animates the slide.
    let tiles = new Map();     // id -> { v, r, c, el }
    let nextTileId = 1;

    // Bookkeeping for the post-slide finalize pass.
    let pendingMerges = [];    // { removeId, upgradeId, newV, r, c }
    let pendingSpawns = [];    // { r, c, v }
    let animTimer = 0;

    // ---------- DOM refs ----------
    const boardEl = document.getElementById('g2048_board');
    const gridEl  = document.getElementById('grid');
    const tilesEl = document.getElementById('tiles');
    const overlayEl = document.getElementById('overlay');
    const overlayTitle = document.getElementById('overlay_title');
    // Modal focus handling: when the overlay opens, focus moves to its action
    // button and is restored to the board on close (see updateOverlay). The
    // overlay holds a single button, so trapping Tab just pins focus to it.
    let overlayWasShown = false;
    let overlayLastFocus = null;
    overlayEl.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') { e.preventDefault(); document.getElementById('overlay_btn').focus(); }
    });
    const scoreEl = document.getElementById('score');
    const bestEl  = document.getElementById('best');
    const undoBtn = document.getElementById('undo_btn');

    // Build static 4x4 background cells once.
    (function buildBg(){
      for (let i = 0; i < SIZE * SIZE; i++) {
        const c = document.createElement('div');
        c.className = 'g2048-cell';
        gridEl.appendChild(c);
      }
    })();

    // ---------- Helpers ----------
    function emptyGrid() {
      const g = new Array(SIZE);
      for (let r = 0; r < SIZE; r++) g[r] = new Array(SIZE).fill(0);
      return g;
    }
    function cloneGrid(g) { return g.map(row => row.slice()); }

    function loadBest() {
      try {
        const v = parseInt(localStorage.getItem(BEST_KEY) || '0', 10);
        best = isFinite(v) && v > 0 ? v : 0;
      } catch(_) { best = 0; }
    }
    function saveBest() {
      try { localStorage.setItem(BEST_KEY, String(best)); } catch(_) {}
    }
    function saveState() {
      try {
        localStorage.setItem(STATE_KEY, JSON.stringify({ grid, score, won, tainted, aiUsed }));
      } catch(_) {}
    }
    function loadState() {
      try {
        const raw = localStorage.getItem(STATE_KEY);
        if (!raw) return false;
        const s = JSON.parse(raw);
        if (!s || !Array.isArray(s.grid) || s.grid.length !== SIZE) return false;
        for (let r = 0; r < SIZE; r++) {
          if (!Array.isArray(s.grid[r]) || s.grid[r].length !== SIZE) return false;
        }
        grid = s.grid;
        score = (typeof s.score === 'number') ? s.score : 0;
        won = !!s.won;
        tainted = !!s.tainted;
        aiUsed = !!s.aiUsed;
        return true;
      } catch(_) { return false; }
    }

    function cellSizePx() {
      const rect = tilesEl.getBoundingClientRect();
      return (rect.width - GAP_PX * (SIZE - 1)) / SIZE;
    }

    // Map (line-index i, slot j) -> (grid row, col) for each slide direction.
    // The line is oriented head-toward-slide, so compaction trivially writes
    // non-zeros to small j.
    function lineCoord(dir, i, j) {
      if (dir === 'left')  return [i, j];
      if (dir === 'right') return [i, SIZE - 1 - j];
      if (dir === 'up')    return [j, i];
      /* down */           return [SIZE - 1 - j, i];
    }

    function findRandomEmpty() {
      const empties = [];
      for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
        if (grid[r][c] === 0) empties.push([r, c]);
      }
      if (empties.length === 0) return null;
      return empties[Math.floor(Math.random() * empties.length)];
    }

    function anyMovePossible() {
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          if (grid[r][c] === 0) return true;
          const v = grid[r][c];
          if (v * 2 > MAX_TILE) continue;   // capped tiles can't merge
          if (r + 1 < SIZE && grid[r + 1][c] === v) return true;
          if (c + 1 < SIZE && grid[r][c + 1] === v) return true;
        }
      }
      return false;
    }

    function maxTile() {
      let m = 0;
      for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
        if (grid[r][c] > m) m = grid[r][c];
      }
      return m;
    }

    // ---------- Tile DOM ----------
    // Outer node owns position (translate + slide transition). Inner node owns
    // appearance and pop/merge keyframes (scale only). They never fight each
    // other for the `transform` property.
    function setTileVisuals(el, v, r, c) {
      const size = cellSizePx();
      const x = c * (size + GAP_PX);
      const y = r * (size + GAP_PX);
      el.style.width  = size + 'px';
      el.style.height = size + 'px';
      el.style.transform = `translate(${x}px, ${y}px)`;
      const inner = el.firstChild;
      const digits = String(v).length;
      const base = size * (digits <= 2 ? 0.45 : digits === 3 ? 0.36 : digits === 4 ? 0.30 : 0.24);
      inner.style.fontSize = Math.max(14, Math.round(base)) + 'px';
      inner.textContent = String(v);
      inner.setAttribute('data-v', String(v));
      if (v > 2048) inner.setAttribute('data-v-big', '');
      else inner.removeAttribute('data-v-big');
    }

    // Update only the outer transform (lets the CSS transition own the slide).
    function setTilePos(el, r, c) {
      const size = cellSizePx();
      const x = c * (size + GAP_PX);
      const y = r * (size + GAP_PX);
      el.style.transform = `translate(${x}px, ${y}px)`;
    }

    function createTileEl(v, r, c, popClass) {
      const el = document.createElement('div');
      el.className = 'g2048-tile' + (popClass ? ' ' + popClass : '');
      const inner = document.createElement('div');
      inner.className = 'g2048-tile-inner';
      el.appendChild(inner);
      // Position BEFORE inserting into the DOM. Otherwise the freshly mounted
      // outer briefly has its initial transform (identity = translate(0,0)) as
      // the previous computed value, and the CSS transition animates from
      // (0,0) to the target — making every new tile slide in from the top-left.
      setTileVisuals(el, v, r, c);
      tilesEl.appendChild(el);
      return el;
    }

    function spawnTileAt(r, c, v, popClass) {
      const id = nextTileId++;
      grid[r][c] = v;
      tileIds[r][c] = id;
      const el = createTileEl(v, r, c, popClass || 'appear');
      tiles.set(id, { v, r, c, el });
      return id;
    }

    function clearAllTiles() {
      tiles.clear();
      while (tilesEl.firstChild) tilesEl.removeChild(tilesEl.firstChild);
    }

    function rebuildTilesFromGrid(popClass) {
      tileIds = emptyGrid();
      clearAllTiles();
      for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
        if (grid[r][c] !== 0) {
          const id = nextTileId++;
          tileIds[r][c] = id;
          const el = createTileEl(grid[r][c], r, c, popClass);
          tiles.set(id, { v: grid[r][c], r, c, el });
        }
      }
    }

    function reflowTiles() {
      // Snap tiles to the new geometry (no slide): a reflow is a resize, not a
      // move. Re-enable the transition next frame so genuine moves still animate.
      tilesEl.classList.add('no-slide');
      for (const t of tiles.values()) setTileVisuals(t.el, t.v, t.r, t.c);
      requestAnimationFrame(() => tilesEl.classList.remove('no-slide'));
    }

    // Previous score shown, so updateScoreDom can pulse the value only when it
    // actually goes up. null on the first render so loading a saved game (score
    // already > 0) doesn't bump; a new game / undo lowers it, which never bumps.
    let lastShownScore = null;
    function bumpScore() {
      scoreEl.classList.remove('bump');
      void scoreEl.offsetWidth;   // restart the keyframe if it's mid-flight
      scoreEl.classList.add('bump');
    }
    function updateScoreDom() {
      scoreEl.textContent = String(score);
      bestEl.textContent  = String(best);
      if (lastShownScore !== null && score > lastShownScore) bumpScore();
      lastShownScore = score;
      refreshUndoBtn();
    }
    // Undo only does something when there's a prior position to restore. History
    // isn't persisted, so it's also empty right after a reload until the first move.
    // Called from updateScoreDom, which every history change (move/undo/new game)
    // already passes through.
    function refreshUndoBtn() {
      if (undoBtn) undoBtn.disabled = history.length === 0;
    }

    function updateOverlay() {
      const note = document.getElementById('overlay_note');
      const show = (overlayMode === 'lose' || overlayMode === 'win');
      if (overlayMode === 'lose') {
        overlayTitle.textContent = t('game_over');
        document.getElementById('overlay_btn').textContent = t('play_again');
        overlayEl.classList.add('show');
      } else if (overlayMode === 'win') {
        overlayTitle.textContent = t('you_win');
        document.getElementById('overlay_btn').textContent = t('keep_going');
        overlayEl.classList.add('show');
      } else {
        overlayEl.classList.remove('show');
      }
      // Move focus into the dialog the first time it appears (keyboard users land
      // on the action button); restore it to the board once it closes. Guarded by
      // overlayWasShown so the many no-op updateOverlay() calls don't steal focus.
      if (show && !overlayWasShown) {
        overlayLastFocus = document.activeElement;
        document.getElementById('overlay_btn').focus();
      } else if (!show && overlayWasShown) {
        const back = (overlayLastFocus && overlayLastFocus.focus) ? overlayLastFocus : boardEl;
        back.focus({ preventScroll: true });
        overlayLastFocus = null;
      }
      overlayWasShown = show;
      // Surface why an AI-assisted run won't appear on the leaderboard. Keyed on
      // aiUsed, not tainted — an undo also taints but the AI note wouldn't fit.
      if (note) { note.innerHTML = withBrand(t('ai_note')); note.hidden = !(overlayMode && aiUsed); }
      if (typeof refreshAiButtons === 'function') refreshAiButtons();
    }

    // ---------- Move / merge with identity ----------
    // Process one line. Returns { newIds, newVals, gained, moves }.
    // moves: per source tile, where it ended up in the output line and whether
    // it merged into a survivor (whose id stays in `newIds`).
    function moveLine(idsLine, valsLine) {
      const newIds = new Array(SIZE).fill(0);
      const newVals = new Array(SIZE).fill(0);
      const moves = [];
      let writeIdx = 0;
      let canMerge = true;
      let gained = 0;
      for (let i = 0; i < SIZE; i++) {
        const id = idsLine[i];
        const v  = valsLine[i];
        if (id === 0) continue;
        // Block the merge that would exceed the ceiling — two MAX_TILE tiles
        // stay separate instead of forming 131072.
        if (writeIdx > 0 && canMerge && newVals[writeIdx - 1] === v && v * 2 <= MAX_TILE) {
          const surviveId = newIds[writeIdx - 1];
          const merged = v * 2;
          gained += merged;
          newVals[writeIdx - 1] = merged;
          moves.push({ id, toIdx: writeIdx - 1, mergedWith: surviveId, newV: merged });
          canMerge = false;
        } else {
          newIds[writeIdx] = id;
          newVals[writeIdx] = v;
          moves.push({ id, toIdx: writeIdx, mergedWith: null });
          writeIdx++;
          canMerge = true;
        }
      }
      return { newIds, newVals, moves, gained };
    }

    function doMove(dir, fromAi = false) {
      if (overlayMode === 'lose' || aiBusy || (aiAuto && !fromAi)) return;
      // Snap any in-flight animation to its committed state before we mutate.
      finalizeAnimation();

      // Pick a slide duration based on how recently the previous input fired.
      // Faster input → shorter slide, so the next move doesn't pile up behind
      // a leisurely transition. The duration is published as a CSS custom
      // property so every tile's transition picks it up automatically.
      const now = performance.now();
      let slideMs = SLIDE_MS_MAX;
      if (lastInputTs > 0) {
        const gap = now - lastInputTs;
        if (gap <= GAP_FAST_MS) {
          slideMs = SLIDE_MS_MIN;
        } else if (gap < GAP_SLOW_MS) {
          const t = (gap - GAP_FAST_MS) / (GAP_SLOW_MS - GAP_FAST_MS);
          slideMs = Math.round(SLIDE_MS_MIN + (SLIDE_MS_MAX - SLIDE_MS_MIN) * t);
        }
      }
      lastInputTs = now;
      tilesEl.style.setProperty('--slide-ms', slideMs + 'ms');

      const snapshot = { grid: cloneGrid(grid), score, won };

      const newGrid = emptyGrid();
      const newIds  = emptyGrid();
      const allMoves = [];   // each: { id, toIdx, mergedWith, newV?, i, dir }
      let totalGained = 0;
      let anyChange = false;

      for (let i = 0; i < SIZE; i++) {
        const idsLine  = new Array(SIZE);
        const valsLine = new Array(SIZE);
        for (let j = 0; j < SIZE; j++) {
          const [r, c] = lineCoord(dir, i, j);
          idsLine[j]  = tileIds[r][c];
          valsLine[j] = grid[r][c];
        }
        const res = moveLine(idsLine, valsLine);
        totalGained += res.gained;
        for (let j = 0; j < SIZE; j++) {
          const [r, c] = lineCoord(dir, i, j);
          newGrid[r][c] = res.newVals[j];
          newIds[r][c]  = res.newIds[j];
          if (idsLine[j] !== res.newIds[j] || valsLine[j] !== res.newVals[j]) anyChange = true;
        }
        for (const m of res.moves) allMoves.push(Object.assign({}, m, { i, dir }));
      }
      if (!anyChange) return;

      // Commit grid state.
      history.push(snapshot);
      if (history.length > UNDO_LIMIT) history.shift();
      grid = newGrid;
      tileIds = newIds;
      score += totalGained;
      if (score > best) { best = score; saveBest(); }

      // Trigger slides. Each existing tile gets its `transform` updated to its
      // destination; the CSS transition handles the actual motion.
      for (const m of allMoves) {
        const [toR, toC] = lineCoord(m.dir, m.i, m.toIdx);
        const tile = tiles.get(m.id);
        if (!tile) continue;
        tile.r = toR;
        tile.c = toC;
        setTilePos(tile.el, toR, toC);
        if (m.mergedWith !== null) {
          // The merging tile slides into the survivor's cell; both DOM nodes
          // overlap momentarily, then the merging one is removed and the
          // survivor pops up with its new value.
          pendingMerges.push({
            removeId: m.id,
            upgradeId: m.mergedWith,
            newV: m.newV,
            r: toR, c: toC,
          });
        }
      }

      // Queue spawn of a new random tile for AFTER the slide. We do NOT write
      // it into `grid` yet — that happens in finalizeAnimation — so a rapid
      // second move that re-slides finds an empty cell where the spawn will
      // land.
      const empty = findRandomEmpty();
      if (empty) {
        const v = Math.random() < 0.9 ? 2 : 4;
        pendingSpawns.push({ r: empty[0], c: empty[1], v });
      }

      updateScoreDom();

      if (animTimer) clearTimeout(animTimer);
      animTimer = setTimeout(() => { animTimer = 0; finalizeAnimation(); }, slideMs + FINALIZE_PAD_MS);
    }

    function finalizeAnimation() {
      if (animTimer) { clearTimeout(animTimer); animTimer = 0; }
      if (pendingMerges.length === 0 && pendingSpawns.length === 0) return;

      for (const m of pendingMerges) {
        const dying = tiles.get(m.removeId);
        if (dying && dying.el && dying.el.parentNode) dying.el.parentNode.removeChild(dying.el);
        tiles.delete(m.removeId);
        const survivor = tiles.get(m.upgradeId);
        if (survivor) {
          survivor.v = m.newV;
          setTileVisuals(survivor.el, survivor.v, survivor.r, survivor.c);
          // Restart the merge keyframe by toggling the class.
          survivor.el.classList.remove('merged');
          void survivor.el.offsetWidth;  // force reflow
          survivor.el.classList.add('merged');
        }
      }
      pendingMerges = [];

      for (const s of pendingSpawns) {
        // The cell should still be empty (snap callers haven't moved between),
        // but be defensive.
        if (grid[s.r][s.c] === 0) spawnTileAt(s.r, s.c, s.v, 'appear');
      }
      pendingSpawns = [];

      if (!won && maxTile() >= 2048) {
        won = true;
        overlayMode = 'win';
      } else if (!anyMovePossible()) {
        overlayMode = 'lose';
      }
      // Report a genuine (un-edited) game's score to the leaderboard. Server
      // keeps the max; SkzLB dedupes, so re-reporting on win→keep-going→lose
      // is harmless.
      if (overlayMode && !tainted && window.SkzLB) window.SkzLB.submit('2048', score);
      saveState();
      updateOverlay();
    }

    function undo() {
      aiSeq++;
      aiBusy = false;
      finalizeAnimation();
      const prev = history.pop();
      if (!prev) return;
      tainted = true;   // undoing a move disqualifies the run from the leaderboard
      grid = prev.grid;
      score = prev.score;
      won = prev.won;
      overlayMode = null;
      rebuildTilesFromGrid(null);  // no pop animation on undo
      updateScoreDom();
      updateOverlay();
      saveState();
      if (aiAuto) setTimeout(requestAiMove, 0);
    }

    function newGame() {
      aiSeq++;
      aiBusy = false;
      if (animTimer) { clearTimeout(animTimer); animTimer = 0; }
      pendingMerges = [];
      pendingSpawns = [];
      grid = emptyGrid();
      tileIds = emptyGrid();
      score = 0;
      won = false;
      history = [];
      overlayMode = null;
      tainted = false;
      aiUsed = false;
      clearAllTiles();
      const e1 = findRandomEmpty();
      if (e1) spawnTileAt(e1[0], e1[1], Math.random() < 0.9 ? 2 : 4, 'appear');
      const e2 = findRandomEmpty();
      if (e2) spawnTileAt(e2[0], e2[1], Math.random() < 0.9 ? 2 : 4, 'appear');
      updateScoreDom();
      updateOverlay();
      if (aiAuto) setTimeout(requestAiMove, 0);
      saveState();
    }

    // ---------- Input ----------
    const KEY_DIR = {
      'ArrowLeft': 'left', 'ArrowRight': 'right',
      'ArrowUp': 'up', 'ArrowDown': 'down',
      'a': 'left', 'A': 'left',
      'd': 'right', 'D': 'right',
      'w': 'up', 'W': 'up',
      's': 'down', 'S': 'down',
    };
    window.addEventListener('keydown', (e) => {
      // Ignore when typing in an input/textarea (none on this page today, but
      // be defensive).
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      const d = KEY_DIR[e.key];
      if (!d) return;
      e.preventDefault();
      doMove(d);
    }, { passive: false });

    // Touch swipe
    let touchStart = null;
    boardEl.addEventListener('touchstart', (e) => {
      if (e.touches.length !== 1) { touchStart = null; return; }
      const t0 = e.touches[0];
      touchStart = { x: t0.clientX, y: t0.clientY };
    }, { passive: true });
    boardEl.addEventListener('touchend', (e) => {
      if (!touchStart) return;
      const t0 = e.changedTouches[0];
      const dx = t0.clientX - touchStart.x;
      const dy = t0.clientY - touchStart.y;
      const adx = Math.abs(dx), ady = Math.abs(dy);
      const THR = 24;
      if (Math.max(adx, ady) < THR) { touchStart = null; return; }
      if (adx > ady) doMove(dx > 0 ? 'right' : 'left');
      else            doMove(dy > 0 ? 'down'  : 'up');
      touchStart = null;
    }, { passive: true });

    // Buttons
    document.getElementById('new_btn').addEventListener('click', newGame);
    document.getElementById('undo_btn').addEventListener('click', undo);
    document.getElementById('overlay_btn').addEventListener('click', () => {
      if (overlayMode === 'win') {
        // "Keep going" — dismiss overlay, don't trigger again until next game.
        overlayMode = null;
        updateOverlay();
      } else {
        newGame();
      }
    });

    // Keep the square board fully inside the viewport. `width:100%` already caps
    // it to the container/viewport width; here we additionally cap `max-width`
    // by the vertical room left below the board's top edge. With aspect-ratio
    // 1/1 the board then shrinks to the SMALLER of (available width, available
    // height) and stays square — so short/landscape windows, and the downward
    // shift when the AI speed slider expands, no longer push it off the bottom.
    // The board's top is fixed by the chrome above it (topbar, scores, actions,
    // slider, status) and does NOT move when the board itself resizes, so this
    // can't feed back into a loop.
    const BOARD_MIN_PX = 200;       // never shrink below a playable size
    const BOARD_BOTTOM_GAP = 16;    // breathing room under the board
    function fitBoard() {
      const top = boardEl.getBoundingClientRect().top;
      const avail = window.innerHeight - top - BOARD_BOTTOM_GAP;
      boardEl.style.maxWidth = Math.max(BOARD_MIN_PX, Math.round(avail)) + 'px';
    }

    // Re-fit + re-position tiles on resize (their pixel coords depend on board
    // width, which may change either from the viewport or from fitBoard's cap).
    let resizeRaf = 0;
    window.addEventListener('resize', () => {
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => { resizeRaf = 0; fitBoard(); reflowTiles(); });
    });

    // ---------- AI (Stochastic Gumbel AlphaZero, browser-side) ----------
    // Loads the exported 2048 net (models/value.onnx) in a Web Worker and plays
    // moves via the full afterstate Stochastic Gumbel AlphaZero MCTS (engine/mcts.js,
    // a port of SkyZero_2048/python/mcts.py): a tree of DECISION + CHANCE nodes,
    // PUCT with MuZero min-max Q-norm, and Gumbel sequential halving at the root.
    // `value` reported back is the search's expected discounted score from here.
    // An AI-assisted run is marked `tainted` so it never posts a score to the
    // human leaderboard.
    // Cache-bust: .onnx is served `immutable, max-age=1yr` (see _headers), so the
    // browser would otherwise never refetch an updated model — a plain reload would
    // keep showing the stale net, and the only way to get the new one would be a
    // "clear site data", which also wipes the leaderboard token in localStorage.
    // Bump AI_MODEL_VERSION whenever models/value.onnx is re-exported.
    const AI_MODEL_VERSION = 'v2-b5c96-vt-iter2335';
    const AI_MODEL_URL = '/2048/models/value.onnx?v=' + AI_MODEL_VERSION;
    // AI speed/strength: the slider sets the wall-clock TIME PER MOVE (ms), not a
    // raw simulation count. Left = faster + weaker (short budget), right = slower
    // + stronger (long budget). Default 330ms ≈ 3 moves/sec. The on-screen cadence
    // is held to exactly the slider's interval by padding (see onAiMove), so
    // "N steps/sec" is honest and device-independent; only the *strength* (how many
    // sims fit in the budget) scales with the machine.
    const AI_STEP_MIN_MS = 50;        // fastest: ~20 moves/sec
    const AI_STEP_MAX_MS = 2000;      // slowest: 0.5 moves/sec
    const AI_STEP_DEFAULT_MS = 330;   // default: ~3 moves/sec
    const AI_STEP_SNAP_MS = 40;       // magnetic detent at the default
    // Reserve part of each interval for the slide animation + settle, so the think
    // budget is interval − reserve and consecutive moves don't visually collide.
    const AI_RENDER_RESERVE_MS = 140;  // capped at ½ the interval — see aiThinkBudgetMs
    const AI_THINK_MIN_MS = 40;        // floor so the fast end still searches a bit
    // Adaptive sim budget: we don't know how fast this device evaluates the net,
    // so we measure ms-per-sim (EWMA) and pick the sim count that fills the think
    // budget. Clamped so the fast end isn't random and the slow end can't run away.
    // Floor at 24 (not 8): below ~24 sims the search is barely past the raw
    // policy, so on slow devices the fast end of the slider would play near-
    // randomly. A slow device that can't fit 24 sims in the budget just drops
    // frames (onAiMove already re-fires immediately on overrun) rather than
    // playing badly. Cap at 2500 (not 1000): the per-move TIME budget is the real
    // governor, so the cap only matters as a guard against runaway sims/memory on
    // very fast machines. 1000 used to pin the slider's whole slow half on a fast
    // box (drag right = slower, no stronger); 2500 leaves that half live and gives
    // headroom for hard late-game positions / a bigger net, while still bounding
    // the waste (b3c64's own strength saturates by ~200-400 sims anyway).
    const AI_SIMS_MIN = 24, AI_SIMS_MAX = 2500, AI_SIMS_BOOTSTRAP = 64;
    const AI_EWMA_ALPHA = 0.3;
    let aiMsPerSim = 0;               // 0 until the first move is measured
    let aiThinkStartTs = 0;           // performance.now() when the think was dispatched
    let aiReady = false, aiBusy = false, aiAuto = false, aiSeq = 0, aiWorker = null;
    const aiAutoBtn = document.getElementById('ai_auto_btn');
    const aiStatusEl = document.getElementById('ai_status');
    const aiSpeedEl = document.getElementById('ai_speed');
    const aiSpeedRow = document.getElementById('ai_speed_row');
    const aiSpeedValEl = document.getElementById('ai_speed_val');

    function setAiStatus(text) { if (aiStatusEl) aiStatusEl.textContent = text || ''; }
    // Loading line carries the SkyZero wordmark, so it goes through innerHTML
    // (withBrand) rather than setAiStatus's textContent. Suffix is numbers only.
    function setAiLoading(suffix) { if (aiStatusEl) aiStatusEl.innerHTML = withBrand(t('ai_loading') + (suffix || '')); }
    function fmtMB(bytes) { return (Number(bytes || 0) / 1048576).toFixed(1); }
    // Current per-move interval (ms) from the slider — read live so dragging it
    // mid-autoplay takes effect on the next move.
    function aiStepMs() {
      const v = aiSpeedEl ? parseInt(aiSpeedEl.value, 10) : AI_STEP_DEFAULT_MS;
      return Math.min(AI_STEP_MAX_MS, Math.max(AI_STEP_MIN_MS, v || AI_STEP_DEFAULT_MS));
    }
    // Live "steps/sec" readout under the slider. The page holds the on-screen
    // cadence to exactly the slider's interval (see onAiMove), so 1000/interval is
    // the honest move rate. Strength (sims/move) scales with the device, not the
    // slider's time budget alone, so it isn't shown here.
    function fmtAiSpeed() {
      const sps = 1000 / aiStepMs();
      const n = sps >= 10 ? String(Math.round(sps)) : String(+sps.toFixed(1));
      return LANG === 'zh' ? '≈ ' + n + ' 步/秒' : '≈ ' + n + ' moves/s';
    }
    function refreshAiSpeed() {
      if (aiSpeedValEl) aiSpeedValEl.textContent = fmtAiSpeed();
    }
    // How long the search may run this move (leave room for the slide + settle).
    function aiThinkBudgetMs() {
      // Reserve shrinks with the interval (capped at half) so the fast end isn't
      // starved — the budget ramps down smoothly instead of slamming into
      // AI_THINK_MIN_MS across the whole fast band. No-op for interval ≥ 280ms
      // (covers the default + slow end), so their budgets are unchanged.
      const reserve = Math.min(AI_RENDER_RESERVE_MS, aiStepMs() / 2);
      return Math.max(AI_THINK_MIN_MS, aiStepMs() - reserve);
    }
    // Sim count that should fill the think budget on this device. Until we've
    // measured the net's per-sim cost, use a modest bootstrap and learn from it.
    function aiPlanSims() {
      if (aiMsPerSim <= 0) return AI_SIMS_BOOTSTRAP;
      const n = Math.round(aiThinkBudgetMs() / aiMsPerSim);
      return Math.min(AI_SIMS_MAX, Math.max(AI_SIMS_MIN, n));
    }

    function refreshAiButtons() {
      if (!aiAutoBtn) return;
      aiAutoBtn.innerHTML = withBrand(aiAuto ? t('ai_stop') : t('ai_auto'));
      aiAutoBtn.setAttribute('aria-pressed', String(aiAuto));
      const gameOver = (overlayMode === 'lose');
      aiAutoBtn.disabled = !aiReady || gameOver;
      // The speed slider is only shown (animated) while autoplay is engaged.
      if (aiSpeedRow) aiSpeedRow.classList.toggle('open', aiAuto);
    }

    function startAiWorker() {
      try {
        aiWorker = new Worker('/2048/worker.js?v=' + Date.now());
      } catch (e) { setAiStatus(t('ai_error')); return; }
      aiWorker.onmessage = (ev) => {
        const m = ev.data || {};
        if (m.type === 'model-progress') {
          if (!aiReady) {
            if (m.percent != null && m.percent >= 100) {
              // Download done — worker is now building the ONNX session (ort WASM
              // + graph compile), which the bar doesn't track. Say so instead of
              // freezing the line at "100% (2.0/2.0 MB)".
              if (aiStatusEl) aiStatusEl.innerHTML = withBrand(t('ai_initializing'));
            } else {
              let s = '';
              if (m.percent != null) s += ' ' + Math.round(m.percent) + '%';
              if (m.total) s += ' (' + fmtMB(m.loaded) + '/' + fmtMB(m.total) + ' MB)';
              else if (m.loaded) s += ' (' + fmtMB(m.loaded) + ' MB)';
              setAiLoading(s);
            }
          }
        } else if (m.type === 'ready') {
          aiReady = true; setAiStatus(''); refreshAiButtons();
        } else if (m.type === 'move') {
          onAiMove(m);
        } else if (m.type === 'error') {
          aiBusy = false; aiAuto = false;
          setAiStatus(t('ai_error') + ': ' + m.message); refreshAiButtons();
        }
      };
      aiWorker.onerror = () => {
        aiBusy = false; aiAuto = false; setAiStatus(t('ai_error')); refreshAiButtons();
      };
      setAiLoading('');
      aiWorker.postMessage({ type: 'init', model: AI_MODEL_URL });
    }

    function requestAiMove() {
      // Bail if autoplay was stopped: a between-move timer or a late start can
      // still reach here after the user hit stop — don't fire another think.
      if (!aiAuto || !aiReady || aiBusy) return;
      // Flush any in-flight slide so the worker sees the settled board (incl.
      // the freshly spawned tile) and any just-reached win/lose state.
      finalizeAnimation();
      if (overlayMode === 'win') { overlayMode = null; updateOverlay(); }
      if (overlayMode === 'lose' || !anyMovePossible()) {
        aiAuto = false; refreshAiButtons(); return;
      }
      // Asking the AI to move disqualifies the run from the human leaderboard
      // (set here, the instant we dispatch, so a SkyZero-controlled run can never submit).
      tainted = true;
      aiUsed = true;
      aiBusy = true; refreshAiButtons();
      aiThinkStartTs = performance.now();
      aiWorker.postMessage({ type: 'think', id: ++aiSeq, grid: cloneGrid(grid), sims: aiPlanSims() });
    }

    function onAiMove(m) {
      if (m.id !== aiSeq) return;
      aiBusy = false;
      // Measure how long this search actually took and fold it into the EWMA of
      // ms-per-sim, so aiPlanSims() converges on the right sim count for this
      // device. Guard against the noisy first eval (ONNX warmup) and any glitch.
      const elapsed = performance.now() - aiThinkStartTs;
      if (!m.terminal && m.sims > 0 && elapsed > 0 && elapsed < 60000) {
        const per = elapsed / m.sims;
        aiMsPerSim = aiMsPerSim > 0 ? aiMsPerSim * (1 - AI_EWMA_ALPHA) + per * AI_EWMA_ALPHA : per;
      }
      if (m.terminal) { aiAuto = false; refreshAiButtons(); return; }
      // Autoplay was stopped while this move was being computed: drop it rather
      // than applying a move the user no longer wants.
      if (!aiAuto) { refreshAiButtons(); return; }
      doMove(m.dir, true);
      refreshAiButtons();
      // Hold the cadence to exactly the slider's interval: the next think fires at
      // (dispatch + interval), so each move occupies one interval regardless of how
      // long the search took. If the search overran the interval (only at the very
      // fast end on a slow device), go again immediately. Read live so dragging the
      // slider mid-autoplay takes effect; requestAiMove re-checks aiAuto, so a stop
      // during the wait is a no-op.
      const wait = Math.max(0, aiStepMs() - (performance.now() - aiThinkStartTs));
      setTimeout(requestAiMove, wait);
    }

    // Handing over to the AI disqualifies the run from the leaderboard (tainted);
    // no modal interrupts play — the end screen notes it when the game finishes.
    aiAutoBtn.addEventListener('click', () => {
      aiAuto = !aiAuto;
      refreshAiButtons();
      if (aiAuto) requestAiMove();
    });
    if (aiSpeedEl && aiSpeedRow) {
      // Place the default-position marker, and snap the slider to the default
      // value when dragged within AI_STEP_SNAP_MS of it (a magnetic detent).
      aiSpeedRow.style.setProperty('--default-pos',
        String((AI_STEP_DEFAULT_MS - AI_STEP_MIN_MS) / (AI_STEP_MAX_MS - AI_STEP_MIN_MS)));
      aiSpeedEl.addEventListener('input', () => {
        const v = parseInt(aiSpeedEl.value, 10);
        if (v !== AI_STEP_DEFAULT_MS && Math.abs(v - AI_STEP_DEFAULT_MS) <= AI_STEP_SNAP_MS) {
          aiSpeedEl.value = String(AI_STEP_DEFAULT_MS);
        }
        refreshAiSpeed();
      });
      refreshAiSpeed();   // populate the readout before the slider first opens
    }
    // Toggling "SkyZero 接管" slides the speed slider open/closed (a ~280ms height
    // animation), which pushes the board down/up. Watch just the slider row —
    // never the board, so there's no resize loop — and re-fit the board on every
    // frame of the slide so it shrinks/grows smoothly to stay in the viewport.
    if (window.ResizeObserver && aiSpeedRow) {
      const ro = new ResizeObserver(() => { fitBoard(); reflowTiles(); });
      ro.observe(aiSpeedRow);
      // Also watch the status row: it collapses to 0 height when idle (see CSS)
      // and re-expands when AI text appears, both of which move the board's top.
      if (aiStatusEl) ro.observe(aiStatusEl);
    }
    startAiWorker();

    // ---------- Boot ----------
    loadBest();
    const hadSavedState = loadState();
    if (!hadSavedState || !grid) {
      grid = emptyGrid();
      tileIds = emptyGrid();
      // Defer initial DOM creation until layout settles so tile sizes are right.
      requestAnimationFrame(() => {
        fitBoard();   // size the board to the viewport before tiles are placed
        const e1 = findRandomEmpty();
        if (e1) spawnTileAt(e1[0], e1[1], Math.random() < 0.9 ? 2 : 4, 'appear');
        const e2 = findRandomEmpty();
        if (e2) spawnTileAt(e2[0], e2[1], Math.random() < 0.9 ? 2 : 4, 'appear');
        updateScoreDom();
        saveState();
      });
    } else {
      // Recreate tile DOM from the loaded grid (after layout settles).
      requestAnimationFrame(() => {
        fitBoard();
        rebuildTilesFromGrid('appear');
        updateScoreDom();
        // overlayMode isn't persisted, so a restored dead board (player reached
        // game-over, then reloaded) would look playable while doMove silently
        // no-ops every input. Re-derive the lose state here. (A 'win' overlay is
        // a dismissible milestone, not a terminal state, so it isn't restored.)
        if (!anyMovePossible()) { overlayMode = 'lose'; updateOverlay(); }
      });
    }
    // Web-font load nudges the chrome height (text reflow), shifting the board's
    // top; re-fit once fonts settle so the initial size is exact.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => { fitBoard(); reflowTiles(); });
    }
    applyLang();
    updateScoreDom();
    updateOverlay();
    boardEl.focus({ preventScroll: true });
  })();
