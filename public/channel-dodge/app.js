(function(){
  'use strict';

  // ---------- i18n (tiny, just for this page) ----------
  const STR = {
    zh: {
      title_tab: 'SkyZero · Dodge',
      brand: 'SkyZero · Dodge',
      score: '分数', best: '最高分',
      start: '开始', pause: '暂停', resume: '继续', restart: '重新开始',
      legend_title: '弹幕类型 · 得分',
      bullet: '子弹', split: '分裂弹', aimed: '追踪弹', cannon: '炮弹',
      bomb: '炸弹', laser: '激光', heal: '血包',
      joy_tip: '按住拖动',
      hint: 'WASD / 方向键移动 · 手机上用下方摇杆 · 躲过的弹幕越多得分越高',
      intro_title: '躲避 Channel Dodge',
      intro_sub: '操控人物躲避四面八方飞来的弹幕。被击中会扣血,一段时间不被命中将自动回血。血量归零即游戏结束 —— 坚持得越久、躲过的弹幕越多,得分越高。',
      paused_title: '已暂停',
      paused_sub: '点击继续返回战斗。',
      over_title: '游戏结束',
      stat_score: '得分', stat_time: '存活', stat_dodged: '躲过弹幕',
      newbest: '新纪录!',
      roster: '形象', draw_own: '自绘形象',
      av_ball: '原版小球', av_custom: '我的形象',
      draw_title: '画一个形象', draw_clear: '清空', draw_cancel: '取消',
      draw_use: '使用', draw_preview: '预览', draw_eraser: '橡皮',
      draw_color: '颜色', draw_brush: '笔刷',
      draw_empty: '画点东西吧',
      ai_play: 'SkyZero 接管', ai_on: 'SkyZero 接管中',
      ai_note: '本局由 SkyZero 接管,未计入排行榜',
    },
    en: {
      title_tab: 'SkyZero · Dodge',
      brand: 'SkyZero · Dodge',
      score: 'Score', best: 'Best',
      start: 'Start', pause: 'Pause', resume: 'Resume', restart: 'Restart',
      legend_title: 'Bullet types · points',
      bullet: 'Bullet', split: 'Split', aimed: 'Homing', cannon: 'Cannonball',
      bomb: 'Bomb', laser: 'Laser', heal: 'Health',
      joy_tip: 'Hold & drag',
      hint: 'WASD / arrow keys to move · use the joystick below on touch · dodge more to score more',
      intro_title: 'Channel Dodge',
      intro_sub: 'Steer your character to dodge bullets flying in from every side. Getting hit costs HP; survive a while without being hit and you slowly heal. When HP hits zero it is game over — the longer you last and the more you dodge, the higher you score.',
      paused_title: 'Paused',
      paused_sub: 'Tap resume to jump back in.',
      over_title: 'Game over',
      stat_score: 'Score', stat_time: 'Survived', stat_dodged: 'Dodged',
      newbest: 'New best!',
      roster: 'Avatar', draw_own: 'Draw your own',
      av_ball: 'Classic ball', av_custom: 'My avatar',
      draw_title: 'Draw your avatar', draw_clear: 'Clear', draw_cancel: 'Cancel',
      draw_use: 'Use', draw_preview: 'Preview', draw_eraser: 'Eraser',
      draw_color: 'Color', draw_brush: 'Brush',
      draw_empty: 'Draw something first',
      ai_play: 'SkyZero autopilot', ai_on: 'SkyZero playing',
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
  // topbar). Used for the autopilot button + the "not counted" hint, which is
  // why those set innerHTML rather than textContent.
  const BRAND_WM = '<span class="brand-wm"><span class="bw-sky">Sky</span><span class="bw-zero">Zero</span></span>';
  const withBrand = (s) => s.replace('SkyZero', BRAND_WM);

  // Cache the resolved accent color; refresh whenever the theme switches.
  let accentColor = '#4493f8';
  function refreshAccent() {
    const c = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    if (c) accentColor = c;
  }

  // ---------- Theme + lang controls (mirror the other pages) ----------
  function applyTheme(mode) {
    const resolved = (mode === 'auto')
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : mode;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themeMode = mode;
    document.querySelectorAll('#theme_seg .seg-btn').forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.theme === mode));
    });
    if (typeof refreshAccent === 'function') refreshAccent();
  }
  (function initThemeSeg(){
    let mode = 'auto';
    try {
      const v = localStorage.getItem('skz_theme');
      if (v === 'light' || v === 'dark') mode = v;
    } catch(_) {}
    applyTheme(mode);
    document.querySelectorAll('#theme_seg .seg-btn').forEach(b => {
      b.addEventListener('click', () => {
        const m = b.dataset.theme;
        try {
          if (m === 'auto') localStorage.removeItem('skz_theme');
          else localStorage.setItem('skz_theme', m);
        } catch(_) {}
        applyTheme(m);
      });
    });
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (document.documentElement.dataset.themeMode === 'auto') applyTheme('auto');
      });
    } catch(_) {}
  })();

  const $ = (id) => document.getElementById(id);
  const els = {
    score: $('score'), best: $('best'),
    lblScore: $('lbl_score'), lblBest: $('lbl_best'),
    startBtn: $('start_btn'), pauseBtn: $('pause_btn'), aiBtn: $('ai_btn'),
    stage: $('stage'), canvas: $('canvas'),
    hpFill: $('hp_fill'), time: $('time'),
    overlay: $('overlay'), overlayTitle: $('overlay_title'),
    overlaySub: $('overlay_sub'), overlayStats: $('overlay_stats'),
    overlayBtn: $('overlay_btn'),
    brand: $('brand_link'), hint: $('hint'),
    legendTitle: $('legend_title'),
    lgBullet: $('lg_bullet'), lgSplit: $('lg_split'), lgAimed: $('lg_aimed'),
    lgCannon: $('lg_cannon'), lgBomb: $('lg_bomb'), lgLaser: $('lg_laser'),
    lgHeal: $('lg_heal'),
    joystick: $('joystick'), joyBase: $('joy_base'),
    joyKnob: $('joy_knob'), joyTip: $('joy_tip'),
    rosterToggle: $('roster_toggle'), rosterLabel: $('roster_label'),
    rosterBody: $('roster_body'), rosterCurrent: $('roster_current'),
    avatarGrid: $('avatar_grid'), drawBtn: $('draw_btn'),
    drawModal: $('draw_modal'), drawTitle: $('draw_title'),
    drawCanvas: $('draw_canvas'), drawPreview: $('draw_preview'),
    drawPreviewLabel: $('draw_preview_label'),
    drawPalette: $('draw_palette'), drawSizes: $('draw_sizes'),
    drawClear: $('draw_clear'), drawCancel: $('draw_cancel'), drawUse: $('draw_use'),
    aiHint: $('ai_hint'),
  };

  function applyLang() {
    document.documentElement.lang = LANG === 'zh' ? 'zh-CN' : 'en';
    document.title = t('title_tab');
    els.lblScore.textContent = t('score');
    els.lblBest.textContent = t('best');
    els.legendTitle.textContent = t('legend_title');
    els.lgBullet.textContent = t('bullet');
    els.lgSplit.textContent = t('split');
    els.lgAimed.textContent = t('aimed');
    els.lgCannon.textContent = t('cannon');
    els.lgBomb.textContent = t('bomb');
    els.lgLaser.textContent = t('laser');
    els.lgHeal.textContent = t('heal');
    els.joyTip.textContent = t('joy_tip');
    els.hint.textContent = t('hint');
    els.rosterLabel.textContent = t('roster');
    els.drawBtn.textContent = t('draw_own');
    els.drawTitle.textContent = t('draw_title');
    els.drawPreviewLabel.textContent = t('draw_preview');
    els.drawClear.textContent = t('draw_clear');
    els.drawCancel.textContent = t('draw_cancel');
    els.drawUse.textContent = t('draw_use');
    els.pauseBtn.textContent = state.mode === 'paused' ? t('resume') : t('pause');
    els.startBtn.textContent = state.mode === 'playing' ? t('restart') : t('start');
    els.aiBtn.innerHTML = withBrand(state.aiOn ? t('ai_on') : t('ai_play'));
    els.aiHint.innerHTML = withBrand(t('ai_note'));
    if (typeof refreshRoster === 'function') refreshRoster();
    renderOverlay();
  }
  (function initLangSeg(){
    document.querySelectorAll('#lang_seg .seg-btn').forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.lang === LANG));
      b.addEventListener('click', () => {
        LANG = b.dataset.lang;
        try { localStorage.setItem('skz_lang', LANG); } catch(_) {}
        document.querySelectorAll('#lang_seg .seg-btn').forEach(o =>
          o.setAttribute('aria-pressed', String(o.dataset.lang === LANG)));
        applyLang();
      });
    });
  })();

  // ============================================================
  //  Game constants
  // ============================================================
  const PLAYER_R = 11;             // collision radius (the real hitbox)
  const AVATAR_R = 17;             // visual radius of the drawn avatar
  const PLAYER_SPEED = 270;        // px / s (in the logical sim arena, below)
  // The simulation always runs in a fixed SIM_W×SIM_H logical arena, identical to
  // the training environment (DodgeSAC/env_dodge.py: 450×600). The canvas is scaled
  // to fit any screen at render time. This keeps the 271-dim observation (which
  // normalizes by POS_NORM=250 / VEL_NORM=300, tuned for 450×600) screen-independent
  // — without it the continuous SAC policy gets clamped or distorted on phones.
  const SIM_W = 450, SIM_H = 600;
  const MAX_HP = 100;
  const REGEN_DELAY = 3.0;         // seconds without a hit before healing
  const REGEN_RATE = 7;            // hp / s
  const HIT_INVULN = 0.6;          // i-frames after a hit (s)

  // Per-type config. `score` is awarded when the projectile is successfully
  // dodged (leaves the arena, or — for lasers — finishes firing) without
  // having hit the player.
  // Every type gets its OWN color so the player can read the threat at a
  // glance. `turn` (rad/s) is the max angular velocity for steering types
  // (0 / absent = flies straight).
  const TYPES = {
    bullet: { score: 1, dmg: 8,  r: 6,  color: '#38bdf8', glow: '#38bdf8' },                 // cyan — basic
    split:  { score: 1, dmg: 8,  r: 6,  color: '#4ade80', glow: '#4ade80' },                 // green — ring/split
    aimed:  { score: 2, dmg: 12, r: 7,  color: '#fb923c', glow: '#fb923c', turn: 1.12 },     // orange — homing (70% of former 1.6 turn rate)
    cannon: { score: 3, dmg: 24, r: 16, color: '#a78bfa', glow: '#a78bfa' },                 // purple — heavy
    bomb:   { score: 4, dmg: 28, r: 11, color: '#facc15', glow: '#facc15' },                 // amber — timed blast
    laser:  { score: 5, dmg: 30, color: '#f43f5e', glow: '#f43f5e' },                        // red — beam
  };

  // Health pickup (not a threat, so not in TYPES). Touch to heal.
  const HEAL_AMOUNT = 25;
  const HEAL_COLOR = '#22c55e';

  // ============================================================
  //  State
  // ============================================================
  const state = {
    mode: 'ready',          // ready | playing | paused | gameover
    W: 0, H: 0,             // logical (CSS px) arena size
    dpr: 1,
    player: { x: 0, y: 0, hp: MAX_HP, sinceHit: 999, invuln: 0 },
    projectiles: [],        // bullets + cannonballs + aimed
    lasers: [],             // laser beams (telegraph + fire)
    bombs: [],              // timed blast bombs (fuse → explosion)
    pickups: [],            // health packs
    score: 0,
    dodged: 0,
    elapsed: 0,             // seconds survived
    spawnTimer: 0,
    ringTimer: 0,
    laserTimer: 0,
    bombTimer: 0,
    pickupTimer: 0,
    recentKinds: [],        // recent primary spawn kinds (anti-streak)
    lastRing: null,         // last split-burst origin (avoid stacking)
    aiOn: false,            // SAC autopilot driving the player
    aiEverUsed: false,      // sticky: did AI ever drive this round? (taints leaderboard)
    lastTs: 0,
    raf: 0,
  };

  let best = 0;
  try { best = parseInt(localStorage.getItem('cd_best') || '0', 10) || 0; } catch(_) {}

  const ctx = els.canvas.getContext('2d');
  refreshAccent();

  // ============================================================
  //  Avatar system — selectable skins for the controllable dot.
  //  The hitbox stays a small circle (PLAYER_R); the avatar is purely
  //  cosmetic and drawn larger, with a small hitbox dot at the center.
  //  The "champion" avatars are original chibi tributes drawn in code
  //  (no official art), recognisable by signature colors/silhouette.
  // ============================================================
  const TAU = Math.PI * 2;

  const AVATARS = [
    { id: 'ball',   zh: '原版小球', en: 'Classic ball' },
    { id: 'teemo',  zh: '提莫',    en: 'Teemo' },
    { id: 'ahri',   zh: '阿狸',    en: 'Ahri' },
    { id: 'lux',    zh: '拉克丝',   en: 'Lux' },
    { id: 'annie',  zh: '安妮',    en: 'Annie' },
    { id: 'jinx',   zh: '金克丝',   en: 'Jinx' },
    { id: 'blitz',  zh: '机器人',   en: 'Blitzcrank' },
    { id: 'custom', zh: '我的形象', en: 'My avatar' },
  ];
  function avatarName(a) { return LANG === 'zh' ? a.zh : a.en; }

  // --- tiny drawing helpers (operate on any 2d context `g`) ---
  function fc(g, x, y, r) { g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill(); }
  function fe(g, x, y, rx, ry, rot) { g.beginPath(); g.ellipse(x, y, rx, ry, rot || 0, 0, TAU); g.fill(); }
  function tri(g, x1, y1, x2, y2, x3, y3) {
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.lineTo(x3, y3); g.closePath(); g.fill();
  }
  function rr(g, x, y, w, h, r) {
    g.beginPath();
    if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h);
    g.fill();
  }
  function star(g, x, y, outer, inner, pts) {
    g.beginPath();
    for (let i = 0; i < pts * 2; i++) {
      const rad = (i % 2 === 0) ? outer : inner;
      const a = (i / (pts * 2)) * TAU - Math.PI / 2;
      const px = x + Math.cos(a) * rad, py = y + Math.sin(a) * rad;
      i === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
    }
    g.closePath(); g.fill();
  }
  function eyes(g, cx, cy, sp, sz, color) {
    g.fillStyle = color || '#241a18';
    fc(g, cx - sp, cy, sz); fc(g, cx + sp, cy, sz);
    g.fillStyle = '#fff';
    fc(g, cx - sp + sz * 0.35, cy - sz * 0.35, sz * 0.32);
    fc(g, cx + sp + sz * 0.35, cy - sz * 0.35, sz * 0.32);
  }

  function drawTeemo(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#3a7d44'; fe(g, cx, cy + 11 * u, 8 * u, 5 * u);            // green collar
    g.fillStyle = '#c79a5e'; fc(g, cx - 9 * u, cy, 3 * u); fc(g, cx + 9 * u, cy, 3 * u); // ears
    g.fillStyle = '#d9b378'; fc(g, cx, cy + 1 * u, 10 * u);                  // face
    eyes(g, cx, cy + 1 * u, 4 * u, 3 * u);
    g.fillStyle = '#7a5a30'; fc(g, cx, cy + 5 * u, 1.4 * u);                 // nose
    g.fillStyle = '#6b4423'; fe(g, cx, cy - 6 * u, 12 * u, 3.4 * u);         // hat brim
    g.beginPath(); g.arc(cx, cy - 6 * u, 7.5 * u, Math.PI, 0); g.fill();     // hat dome
    g.fillStyle = '#caa94e'; fc(g, cx, cy - 8 * u, 2.2 * u);                 // hat badge
  }
  function drawAhri(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#e8843c'; fe(g, cx + 8 * u, cy + 9 * u, 5 * u, 3 * u, -0.6); // tail
    g.fillStyle = '#fff'; fc(g, cx + 11 * u, cy + 11 * u, 2 * u);
    g.fillStyle = '#e9eef5'; fc(g, cx, cy + 1 * u, 11 * u);                  // silver hair
    g.fillStyle = '#e9eef5';                                                // fox ears
    tri(g, cx - 7 * u, cy - 6 * u, cx - 4 * u, cy - 12 * u, cx - 1 * u, cy - 6 * u);
    tri(g, cx + 7 * u, cy - 6 * u, cx + 4 * u, cy - 12 * u, cx + 1 * u, cy - 6 * u);
    g.fillStyle = '#e8843c';
    tri(g, cx - 5 * u, cy - 7 * u, cx - 4 * u, cy - 10.5 * u, cx - 2.5 * u, cy - 7 * u);
    tri(g, cx + 5 * u, cy - 7 * u, cx + 4 * u, cy - 10.5 * u, cx + 2.5 * u, cy - 7 * u);
    g.fillStyle = '#f6e4d6'; fc(g, cx, cy + 2 * u, 8 * u);                   // pale face
    g.fillStyle = '#caa12e'; fc(g, cx - 3.3 * u, cy + 2 * u, 2.2 * u); fc(g, cx + 3.3 * u, cy + 2 * u, 2.2 * u);
    g.fillStyle = '#2a1f14'; fc(g, cx - 3.3 * u, cy + 2 * u, 1 * u); fc(g, cx + 3.3 * u, cy + 2 * u, 1 * u);
    g.strokeStyle = '#d98f6a'; g.lineWidth = 1 * u; g.lineCap = 'round';     // whiskers
    g.beginPath();
    g.moveTo(cx - 6 * u, cy + 4 * u); g.lineTo(cx - 3 * u, cy + 4.5 * u);
    g.moveTo(cx + 6 * u, cy + 4 * u); g.lineTo(cx + 3 * u, cy + 4.5 * u);
    g.stroke();
  }
  function drawLux(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#ffe680'; star(g, cx + 9 * u, cy - 8 * u, 4 * u, 1.8 * u, 5); // wand star
    g.fillStyle = '#f2d06b'; fc(g, cx, cy + 2 * u, 11 * u);                  // blonde hair
    fe(g, cx - 8 * u, cy + 6 * u, 3 * u, 6 * u); fe(g, cx + 8 * u, cy + 6 * u, 3 * u, 6 * u);
    g.fillStyle = '#f8e6d4'; fc(g, cx, cy + 2 * u, 8 * u);                   // face
    g.fillStyle = '#f2d06b'; g.beginPath(); g.arc(cx, cy - 3 * u, 8 * u, Math.PI, 0); g.fill(); // bangs
    g.fillStyle = '#3a7bd5'; fc(g, cx - 3.2 * u, cy + 3 * u, 2 * u); fc(g, cx + 3.2 * u, cy + 3 * u, 2 * u);
    g.fillStyle = '#fff'; fc(g, cx - 2.6 * u, cy + 2.4 * u, 0.7 * u); fc(g, cx + 3.8 * u, cy + 2.4 * u, 0.7 * u);
    g.strokeStyle = '#c98f6a'; g.lineWidth = 1 * u;
    g.beginPath(); g.arc(cx, cy + 5 * u, 2 * u, 0.2, Math.PI - 0.2); g.stroke();
  }
  function drawAnnie(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#ff7a18'; fc(g, cx + 9 * u, cy + 7 * u, 3.4 * u);         // fireball
    g.fillStyle = '#ffd24a'; fc(g, cx + 9 * u, cy + 7 * u, 1.8 * u);
    g.fillStyle = '#7a2e1e'; fc(g, cx, cy + 1 * u, 11 * u);                  // auburn hair
    fc(g, cx - 10 * u, cy + 1 * u, 3.5 * u); fc(g, cx + 10 * u, cy + 1 * u, 3.5 * u); // pigtails
    g.fillStyle = '#f7e0cf'; fc(g, cx, cy + 2.5 * u, 8 * u);                 // face
    g.fillStyle = '#7a2e1e'; g.beginPath(); g.arc(cx, cy - 2 * u, 8 * u, Math.PI, 0); g.fill(); // bangs
    g.fillStyle = '#2fae6a'; fc(g, cx - 3.3 * u, cy + 3 * u, 2.4 * u); fc(g, cx + 3.3 * u, cy + 3 * u, 2.4 * u);
    g.fillStyle = '#1d130e'; fc(g, cx - 3.3 * u, cy + 3 * u, 1 * u); fc(g, cx + 3.3 * u, cy + 3 * u, 1 * u);
  }
  function drawJinx(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#5ea0e0'; fe(g, cx - 9 * u, cy + 8 * u, 3 * u, 8 * u); fe(g, cx + 9 * u, cy + 8 * u, 3 * u, 8 * u); // braids
    g.fillStyle = '#4a8fd0'; fc(g, cx - 9 * u, cy + 14 * u, 2.6 * u); fc(g, cx + 9 * u, cy + 14 * u, 2.6 * u);
    g.fillStyle = '#5ea0e0'; fc(g, cx, cy + 1 * u, 10.5 * u);                // hair top
    g.fillStyle = '#eadbc8'; fc(g, cx, cy + 2.5 * u, 8 * u);                 // face
    g.fillStyle = '#5ea0e0'; g.fillRect(cx - 8 * u, cy - 6 * u, 16 * u, 5 * u); // straight bangs
    g.fillStyle = '#e0489a'; fc(g, cx - 3.3 * u, cy + 3 * u, 2 * u); fc(g, cx + 3.3 * u, cy + 3 * u, 2 * u);
    g.fillStyle = '#fff'; fc(g, cx - 2.7 * u, cy + 2.5 * u, 0.6 * u); fc(g, cx + 3.9 * u, cy + 2.5 * u, 0.6 * u);
    g.strokeStyle = '#b06b50'; g.lineWidth = 1 * u; g.lineCap = 'round';
    g.beginPath(); g.arc(cx, cy + 5.5 * u, 2.4 * u, 0.15, Math.PI - 0.15); g.stroke();
  }
  function drawBlitz(g, cx, cy, R) {
    const u = R / 16;
    g.fillStyle = '#9a7b2e'; fe(g, cx, cy + 11 * u, 10 * u, 5 * u);          // shoulders
    g.fillStyle = '#caa94e'; rr(g, cx - 9 * u, cy - 9 * u, 18 * u, 16 * u, 5 * u); // gold head
    g.fillStyle = '#2b2b30'; rr(g, cx - 6.5 * u, cy - 4 * u, 13 * u, 8 * u, 3 * u); // face plate
    g.fillStyle = '#48d6ff'; fc(g, cx - 3 * u, cy, 1.8 * u); fc(g, cx + 3 * u, cy, 1.8 * u); // eyes
    g.strokeStyle = '#9a7b2e'; g.lineWidth = 1.6 * u;
    g.beginPath(); g.moveTo(cx, cy - 9 * u); g.lineTo(cx, cy - 13 * u); g.stroke(); // antenna
    g.fillStyle = '#e0483b'; fc(g, cx, cy - 13 * u, 2 * u);
    g.fillStyle = '#8a6f28'; fc(g, cx - 7 * u, cy - 6 * u, 1 * u); fc(g, cx + 7 * u, cy - 6 * u, 1 * u);
  }
  const CHAMP = { teemo: drawTeemo, ahri: drawAhri, lux: drawLux, annie: drawAnnie, jinx: drawJinx, blitz: drawBlitz };

  // Custom (player-drawn) avatar, persisted as a small pixel-art data URL.
  let avatarId = 'ball';
  let customImg = null, customURL = null;
  try { const v = localStorage.getItem('cd_avatar'); if (v) avatarId = v; } catch (_) {}
  try { customURL = localStorage.getItem('cd_avatar_custom') || null; } catch (_) {}
  if (customURL) {
    customImg = new Image();
    customImg.onload = () => { buildRoster(); if (state.mode !== 'playing') draw(); };
    customImg.src = customURL;
  }
  if (!AVATARS.some(a => a.id === avatarId)) avatarId = 'ball';
  if (avatarId === 'custom' && !customURL) avatarId = 'ball';

  // Draw avatar `id` centered at (cx,cy), sized to visual radius R.
  function drawAvatar(g, id, cx, cy, R) {
    if (id === 'custom') {
      if (customImg && customImg.complete && customImg.naturalWidth) {
        // 1:1 drawing scaled smoothly into the avatar box.
        g.drawImage(customImg, cx - R, cy - R, R * 2, R * 2);
        return;
      }
      id = 'ball';
    }
    const fn = CHAMP[id];
    if (fn) { fn(g, cx, cy, R); return; }
    g.fillStyle = accentColor; fc(g, cx, cy, R * 0.72);            // default ball
    g.fillStyle = 'rgba(255,255,255,0.9)'; fc(g, cx, cy, R * 0.28);
  }

  // ---------- roster UI ----------
  function buildRoster() {
    const grid = els.avatarGrid;
    grid.innerHTML = '';
    const muted = getComputedStyle(document.documentElement).getPropertyValue('--fg-subtle').trim() || '#888';
    AVATARS.forEach((a, idx) => {
      const empty = a.id === 'custom' && !(customImg && customImg.complete && customImg.naturalWidth);
      const btn = document.createElement('button');
      btn.className = 'cd-avatar-btn';
      btn.type = 'button';
      btn.setAttribute('aria-pressed', String(a.id === avatarId));
      const c = document.createElement('canvas');
      c.width = 48; c.height = 48;
      const g = c.getContext('2d');
      if (empty) {
        g.strokeStyle = muted; g.lineWidth = 2.5; g.lineCap = 'round';
        g.beginPath(); g.moveTo(24, 15); g.lineTo(24, 33); g.moveTo(15, 24); g.lineTo(33, 24); g.stroke();
      } else {
        drawAvatar(g, a.id, 24, 25, 19);
      }
      const label = document.createElement('span');
      label.textContent = empty ? t('draw_own') : avatarName(a);
      btn.appendChild(c); btn.appendChild(label);
      btn.addEventListener('click', () => {
        if (empty) { openDraw(); return; }
        selectAvatar(a.id);
      });
      grid.appendChild(btn);
    });
    updateRosterCurrent();
  }
  function refreshRoster() { buildRoster(); }   // rebuild on language change
  function updateRosterCurrent() {
    const a = AVATARS.find(x => x.id === avatarId) || AVATARS[0];
    els.rosterCurrent.textContent = avatarName(a);
  }
  function selectAvatar(id) {
    avatarId = id;
    try { localStorage.setItem('cd_avatar', id); } catch (_) {}
    els.avatarGrid.querySelectorAll('.cd-avatar-btn').forEach((b, i) => {
      b.setAttribute('aria-pressed', String(AVATARS[i].id === id));
    });
    updateRosterCurrent();
    if (state.mode !== 'playing') draw();
  }
  els.rosterToggle.addEventListener('click', () => {
    const closed = els.rosterBody.hasAttribute('hidden');
    els.rosterBody.toggleAttribute('hidden', !closed);
    els.rosterToggle.setAttribute('aria-expanded', String(closed));
  });
  els.drawBtn.addEventListener('click', openDraw);

  // ---------- draw-your-own editor ----------
  const PALETTE = ['#1a1a1a', '#ffffff', '#e0483b', '#f5a623', '#f8e71c',
                   '#3aa757', '#3a7bd5', '#9b59b6', '#e8843c', '#7a4a24', '#ff6fae', '#48d6ff'];
  const dctx = els.drawCanvas.getContext('2d', { willReadFrequently: true });
  let drawColor = PALETTE[0], drawErase = false, brush = 14;
  let drawing = false, lastX = 0, lastY = 0;

  function buildTools() {
    els.drawPalette.innerHTML = '';
    for (const col of PALETTE) {
      const b = document.createElement('button');
      b.className = 'cd-swatch'; b.type = 'button'; b.style.background = col; b._col = col;
      b.setAttribute('aria-label', t('draw_color') + ' ' + col);
      b.addEventListener('click', () => { drawColor = col; drawErase = false; updateToolUi(); });
      els.drawPalette.appendChild(b);
    }
    const er = document.createElement('button');
    er.className = 'cd-swatch eraser'; er.type = 'button'; er.textContent = '⌫'; er._erase = true;
    er.setAttribute('aria-label', t('draw_eraser'));
    er.addEventListener('click', () => { drawErase = true; updateToolUi(); });
    els.drawPalette.appendChild(er);
    els.drawSizes.innerHTML = '';
    for (const s of [8, 14, 24]) {
      const b = document.createElement('button');
      b.className = 'cd-size-btn'; b.type = 'button'; b._size = s;
      b.setAttribute('aria-label', t('draw_brush') + ' ' + s);
      const dot = document.createElement('span');
      dot.className = 'dot';
      const dd = Math.max(4, Math.round(s / 2.2));
      dot.style.width = dd + 'px'; dot.style.height = dd + 'px';
      b.appendChild(dot);
      b.addEventListener('click', () => { brush = s; updateToolUi(); });
      els.drawSizes.appendChild(b);
    }
    updateToolUi();
  }
  function updateToolUi() {
    els.drawPalette.querySelectorAll('.cd-swatch').forEach(b => {
      b.setAttribute('aria-pressed', String(b._erase ? drawErase : (!drawErase && b._col === drawColor)));
    });
    els.drawSizes.querySelectorAll('.cd-size-btn').forEach(b => {
      b.setAttribute('aria-pressed', String(b._size === brush));
    });
  }
  function dpt(e) {
    const r = els.drawCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (els.drawCanvas.width / r.width),
      y: (e.clientY - r.top) * (els.drawCanvas.height / r.height),
    };
  }
  function applyBrush() {
    dctx.lineCap = 'round'; dctx.lineJoin = 'round'; dctx.lineWidth = brush;
    if (drawErase) { dctx.globalCompositeOperation = 'destination-out'; dctx.strokeStyle = '#000'; dctx.fillStyle = '#000'; }
    else { dctx.globalCompositeOperation = 'source-over'; dctx.strokeStyle = drawColor; dctx.fillStyle = drawColor; }
  }
  els.drawCanvas.addEventListener('pointerdown', (e) => {
    drawing = true; const p = dpt(e); lastX = p.x; lastY = p.y;
    applyBrush(); dctx.beginPath(); dctx.arc(p.x, p.y, brush / 2, 0, TAU); dctx.fill();
    els.drawCanvas.setPointerCapture(e.pointerId); e.preventDefault();
  });
  els.drawCanvas.addEventListener('pointermove', (e) => {
    if (!drawing) return; const p = dpt(e);
    applyBrush(); dctx.beginPath(); dctx.moveTo(lastX, lastY); dctx.lineTo(p.x, p.y); dctx.stroke();
    lastX = p.x; lastY = p.y; e.preventDefault();
  });
  function endDraw() { if (!drawing) return; drawing = false; updatePreview(); }
  els.drawCanvas.addEventListener('pointerup', endDraw);
  els.drawCanvas.addEventListener('pointercancel', endDraw);

  // The drawing is used 1:1 (no pixelation) — we just capture the canvas as-is.
  function drawingNonEmpty() {
    const d = dctx.getImageData(0, 0, els.drawCanvas.width, els.drawCanvas.height).data;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) return true;
    return false;
  }
  function updatePreview() {
    const pg = els.drawPreview.getContext('2d');
    pg.clearRect(0, 0, els.drawPreview.width, els.drawPreview.height);
    pg.imageSmoothingEnabled = true;
    pg.drawImage(els.drawCanvas, 0, 0, els.drawPreview.width, els.drawPreview.height);
  }
  let drawLastFocus = null;
  function openDraw() {
    dctx.setTransform(1, 0, 0, 1, 0, 0);
    dctx.clearRect(0, 0, els.drawCanvas.width, els.drawCanvas.height);
    drawErase = false; drawColor = PALETTE[0]; brush = 14;
    updateToolUi(); updatePreview();
    els.drawPreviewLabel.textContent = t('draw_preview');
    drawLastFocus = document.activeElement;
    els.drawModal.removeAttribute('hidden');
    // Move focus into the dialog (first palette swatch) for keyboard users.
    const first = els.drawModal.querySelector('button');
    if (first) first.focus();
  }
  function closeDraw() {
    els.drawModal.setAttribute('hidden', '');
    if (drawLastFocus && drawLastFocus.focus) drawLastFocus.focus();
    drawLastFocus = null;
  }
  // Keep keyboard focus inside the modal while it's open, and let Esc dismiss it
  // (stopPropagation so the global Esc handler doesn't also pause the game).
  els.drawModal.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeDraw(); return; }
    if (e.key !== 'Tab') return;
    const f = Array.prototype.filter.call(
      els.drawModal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
      (el) => !el.disabled && el.offsetParent !== null);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
  els.drawClear.addEventListener('click', () => {
    dctx.clearRect(0, 0, els.drawCanvas.width, els.drawCanvas.height);
    els.drawPreviewLabel.textContent = t('draw_preview');
    updatePreview();
  });
  els.drawCancel.addEventListener('click', closeDraw);
  els.drawUse.addEventListener('click', () => {
    if (!drawingNonEmpty()) { els.drawPreviewLabel.textContent = t('draw_empty'); return; }
    customURL = els.drawCanvas.toDataURL();   // 1:1 capture, transparent background preserved
    try { localStorage.setItem('cd_avatar_custom', customURL); } catch (_) {}
    customImg = new Image();
    customImg.onload = () => { buildRoster(); selectAvatar('custom'); };
    customImg.src = customURL;
    closeDraw();
  });

  buildTools();
  buildRoster();

  // ---------- responsive canvas ----------
  function resize() {
    const rect = els.stage.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.dpr = dpr;
    state.W = SIM_W;                 // physics live in the fixed logical arena…
    state.H = SIM_H;
    els.canvas.width = Math.round(rect.width * dpr);
    els.canvas.height = Math.round(rect.height * dpr);
    // …rendered letterboxed: scale the fixed 450×600 arena UNIFORMLY to fit the
    // (any-aspect, full-width) canvas and center it — never distorts, never overflows.
    const s = Math.min(els.canvas.width / SIM_W, els.canvas.height / SIM_H);
    state._ox = (els.canvas.width - SIM_W * s) / 2;
    state._oy = (els.canvas.height - SIM_H * s) / 2;
    ctx.setTransform(s, 0, 0, s, state._ox, state._oy);
    // Keep the player inside the (possibly resized) arena.
    const p = state.player;
    p.x = clamp(p.x, PLAYER_R, state.W - PLAYER_R);
    p.y = clamp(p.y, PLAYER_R, state.H - PLAYER_R);
    if (state.mode === 'ready' || state.mode === 'gameover') {
      p.x = state.W / 2; p.y = state.H / 2;
    }
    if (state.mode !== 'playing') draw();
  }
  if (window.ResizeObserver) new ResizeObserver(resize).observe(els.stage);
  window.addEventListener('resize', resize);

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  // Deterministic-free randomness is fine here (gameplay, not resumable).
  function rand(a, b) { return a + Math.random() * (b - a); }

  const AI = SkyZeroDodgeAI.createPolicyController({
    policy: window.DODGE_POLICY,
    getState: () => state,
    types: TYPES,
    maxHp: MAX_HP,
    hitInvuln: HIT_INVULN,
  });
  // ============================================================
  //  Input — keyboard + floating touch joystick
  // ============================================================
  const keys = new Set();
  const MOVE_KEYS = {
    ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
    w: 'up', s: 'down', a: 'left', d: 'right',
    W: 'up', S: 'down', A: 'left', D: 'right',
  };
  window.addEventListener('keydown', (e) => {
    if (e.key in MOVE_KEYS) { keys.add(MOVE_KEYS[e.key]); e.preventDefault(); }
    else if (e.key === ' ' || e.key === 'Enter') {
      if (state.mode === 'ready' || state.mode === 'gameover') { startGame(); e.preventDefault(); }
      else if (state.mode === 'playing') { pauseGame(); e.preventDefault(); }
      else if (state.mode === 'paused') { resumeGame(); e.preventDefault(); }
    } else if (e.key === 'Escape' && state.mode === 'playing') { pauseGame(); }
  });
  window.addEventListener('keyup', (e) => {
    if (e.key in MOVE_KEYS) keys.delete(MOVE_KEYS[e.key]);
  });

  // Touch joystick lives in the dedicated region BELOW the arena, so the
  // thumb never covers the play field. A press anywhere in the region drops
  // the base at that point (floating), and dragging from there sets the move
  // vector. Only wired up on coarse-pointer (touch) devices.
  const joy = { active: false, id: null, dx: 0, dy: 0 };
  const JOY_MAX = 50;
  const isTouch = (window.matchMedia && window.matchMedia('(pointer: coarse)').matches)
                  || ('ontouchstart' in window);

  if (isTouch) {
    document.body.classList.add('cd-touch');
    const region = els.joystick, base = els.joyBase, knob = els.joyKnob;
    const regionPoint = (e) => {
      const r = region.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const setKnob = (dx, dy) => {
      knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
    };
    region.addEventListener('pointerdown', (e) => {
      const p = regionPoint(e);
      joy.active = true; joy.id = e.pointerId; joy.dx = 0; joy.dy = 0;
      // Anchor the base where the thumb landed.
      base.style.left = p.x + 'px';
      base.style.top = p.y + 'px';
      setKnob(0, 0);
      region.classList.add('dragging');
      region.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    region.addEventListener('pointermove', (e) => {
      if (!joy.active || e.pointerId !== joy.id) return;
      const p = regionPoint(e);
      const r = region.getBoundingClientRect();
      // Base center in region coords = the px left/top we set on pointerdown.
      const bx = parseFloat(base.style.left) || r.width / 2;
      const by = parseFloat(base.style.top) || r.height / 2;
      let dx = p.x - bx, dy = p.y - by;
      const len = Math.hypot(dx, dy);
      if (len > JOY_MAX) { dx = dx / len * JOY_MAX; dy = dy / len * JOY_MAX; }
      joy.dx = dx; joy.dy = dy;
      setKnob(dx, dy);
      e.preventDefault();
    });
    const endJoy = (e) => {
      if (e.pointerId !== joy.id) return;
      joy.active = false; joy.id = null; joy.dx = 0; joy.dy = 0;
      // Snap the base back to its resting (centered) position.
      base.style.left = ''; base.style.top = '';
      setKnob(0, 0);
      region.classList.remove('dragging');
    };
    region.addEventListener('pointerup', endJoy);
    region.addEventListener('pointercancel', endJoy);
  }

  function moveVector() {
    if (state.aiOn) return AI.move;                 // SAC autopilot drives
    // Joystick takes priority when held.
    if (joy.active && (joy.dx || joy.dy)) {
      const len = Math.hypot(joy.dx, joy.dy);
      if (len < 6) return { x: 0, y: 0 };           // small dead-zone
      const mag = Math.min(1, len / JOY_MAX);
      return { x: (joy.dx / len) * mag, y: (joy.dy / len) * mag };
    }
    let x = 0, y = 0;
    if (keys.has('left')) x -= 1;
    if (keys.has('right')) x += 1;
    if (keys.has('up')) y -= 1;
    if (keys.has('down')) y += 1;
    if (x && y) { const n = Math.SQRT1_2; x *= n; y *= n; }   // normalize diagonals
    return { x, y };
  }

  // ============================================================
  //  Spawning
  // ============================================================
  // Spawn a point just outside a random edge of the arena.
  function edgeSpawn() {
    const m = 24;
    const side = Math.floor(rand(0, 4));
    if (side === 0) return { x: rand(0, state.W), y: -m };               // top
    if (side === 1) return { x: state.W + m, y: rand(0, state.H) };      // right
    if (side === 2) return { x: rand(0, state.W), y: state.H + m };      // bottom
    return { x: -m, y: rand(0, state.H) };                              // left
  }

  function spawnBullet(speedMul) {
    const s = edgeSpawn();
    // Aim toward a point near the player with some scatter.
    const p = state.player;
    const ang = Math.atan2(p.y - s.y, p.x - s.x) + rand(-0.5, 0.5);
    const spd = rand(150, 230) * speedMul;
    state.projectiles.push({
      kind: 'bullet', x: s.x, y: s.y,
      vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd,
      r: TYPES.bullet.r, scored: false, hitOnce: false,
    });
  }

  function spawnAimed(speedMul) {
    const s = edgeSpawn();
    const p = state.player;
    const ang = Math.atan2(p.y - s.y, p.x - s.x) + rand(-0.08, 0.08);
    const spd = rand(190, 260) * speedMul;
    state.projectiles.push({
      kind: 'aimed', x: s.x, y: s.y,
      vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd,
      r: TYPES.aimed.r, scored: false, hitOnce: false, life: 6,
    });
  }

  function spawnCannon(speedMul) {
    const s = edgeSpawn();
    const p = state.player;
    const ang = Math.atan2(p.y - s.y, p.x - s.x) + rand(-0.2, 0.2);
    const spd = rand(95, 140) * speedMul;
    state.projectiles.push({
      kind: 'cannon', x: s.x, y: s.y,
      vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd,
      r: TYPES.cannon.r, scored: false, hitOnce: false,
    });
  }

  // A radial burst from a point inside the arena — classic bullet-hell ring.
  // These are the green "split" bullets (distinct from the cyan stream).
  function spawnRing(speedMul) {
    // Pick a burst origin that isn't right on top of the previous one, so
    // consecutive split bursts don't stack in the same spot.
    let cx, cy;
    const minSep = Math.min(state.W, state.H) * 0.28;
    for (let tries = 0; tries < 6; tries++) {
      cx = rand(state.W * 0.2, state.W * 0.8);
      cy = rand(state.H * 0.15, state.H * 0.5);
      if (!state.lastRing || Math.hypot(cx - state.lastRing.x, cy - state.lastRing.y) >= minSep) break;
    }
    state.lastRing = { x: cx, y: cy };
    const n = Math.floor(rand(8, 14));
    const spd = rand(120, 170) * speedMul;
    const off = rand(0, Math.PI * 2);
    for (let i = 0; i < n; i++) {
      const ang = off + (i / n) * Math.PI * 2;
      state.projectiles.push({
        kind: 'split', x: cx, y: cy,
        vx: Math.cos(ang) * spd, vy: Math.sin(ang) * spd,
        r: TYPES.split.r, scored: false, hitOnce: false,
      });
    }
  }

  // A timed bomb: lands inside the arena, shows its fuse + blast radius as a
  // telegraph, then detonates. Anything (the player) inside the radius at the
  // moment of detonation takes the hit; survive the blast to score it.
  const BOMB_FUSE = 1.8;        // telegraph seconds
  const BOMB_FLASH = 0.2;       // explosion visible window (s)
  function spawnBomb() {
    const m = 56;
    state.bombs.push({
      x: rand(m, state.W - m),
      y: rand(m, state.H - m),
      blastR: clamp(Math.min(state.W, state.H) * 0.30, 70, 150),
      fuse: BOMB_FUSE,
      flash: 0,
      phase: 'fuse',
      scored: false, hitOnce: false,
    });
  }

  // A health pack: drifts onto the map and heals on contact. Despawns after a
  // while (blinking near the end) if the player doesn't grab it.
  const PICKUP_R = 13;
  const PICKUP_LIFE = 9;        // seconds before it fades away
  function spawnPickup() {
    const m = 36;
    state.pickups.push({
      x: rand(m, state.W - m),
      y: rand(m, state.H - m),
      life: PICKUP_LIFE,
    });
  }

  // A laser: charges (telegraph) along a line through the arena, then fires
  // a thick damaging beam for a short window.
  function spawnLaser() {
    const s = edgeSpawn();
    const p = state.player;
    // Aim through the player's current position; lock direction at spawn so
    // the telegraph is a fair warning the player can dodge out of.
    const ang = Math.atan2(p.y - s.y, p.x - s.x);
    state.lasers.push({
      ox: s.x, oy: s.y, ang,
      charge: 1.0,        // telegraph seconds remaining
      fire: 0.55,         // active beam seconds
      phase: 'charge',
      halfWidth: 13,
      scored: false, hitOnce: false,
    });
  }

  // Director: scale spawn frequency and unlock harder patterns over time.
  function difficulty() {
    // Starts at 1, grows ~linearly but gently; soft-capped. A larger divisor
    // means a slower ramp — the player has more time at each intensity.
    return Math.min(1 + state.elapsed / 50, 3.8);
  }

  // Choose the kind for a primary spawn, then break up long runs of the same
  // kind so the screen doesn't end up flooded with one bullet type. The base
  // roll keeps the original relative mix (ring is handled on its own timer).
  function pickSpawnKind(d, roll) {
    let kind;
    if (d > 1.5 && roll < 0.23) kind = 'cannon';
    else if (roll < 0.56) kind = 'aimed';
    else kind = 'bullet';
    const rk = state.recentKinds;
    // No more than 2 of the same kind in a row.
    if (rk.length >= 2 && rk[rk.length - 1] === kind && rk[rk.length - 2] === kind) {
      const alts = ['bullet', 'aimed', 'cannon']
        .filter(k => k !== kind && (k !== 'cannon' || d > 1.5));
      kind = alts[Math.floor(Math.random() * alts.length)];
    }
    rk.push(kind);
    if (rk.length > 4) rk.shift();
    return kind;
  }

  function runDirector(dt) {
    const d = difficulty();
    const speedMul = 0.9 + d * 0.12;

    state.spawnTimer -= dt;
    if (state.spawnTimer <= 0) {
      // Interval shrinks as difficulty rises.
      state.spawnTimer = rand(0.55, 1.0) / (0.7 + d * 0.4);
      const kind = pickSpawnKind(d, Math.random());
      if (kind === 'cannon') spawnCannon(speedMul);
      else if (kind === 'aimed') spawnAimed(speedMul);
      else spawnBullet(speedMul);
      // Extra simple bullets fired alongside at higher difficulty.
      if (d > 1.8 && Math.random() < 0.5) spawnBullet(speedMul);
      if (d > 3.0 && Math.random() < 0.5) spawnBullet(speedMul);
    }

    // Split bursts fire on their own evenly-spaced cadence (rather than a
    // random per-tick roll, which clustered several bursts back-to-back).
    // The interval is tuned to keep the same average rate as the old roll.
    if (d > 2.2) {
      state.ringTimer -= dt;
      if (state.ringTimer <= 0) {
        state.ringTimer = rand(2.6, 3.6) / (0.55 + d * 0.18);
        spawnRing(speedMul);
      }
    }

    // Lasers unlock a bit later and fire on their own cadence.
    if (d > 1.6) {
      state.laserTimer -= dt;
      if (state.laserTimer <= 0) {
        state.laserTimer = rand(3.2, 5.5) / (0.6 + d * 0.18);
        spawnLaser();
      }
    }

    // Timed bombs unlock mid-game and detonate on their own fuse.
    if (d > 1.4) {
      state.bombTimer -= dt;
      if (state.bombTimer <= 0) {
        state.bombTimer = rand(4.5, 7.5) / (0.7 + d * 0.12);
        spawnBomb();
      }
    }

    // Health packs appear on a slow, difficulty-independent cadence; cap how
    // many sit on the map at once.
    state.pickupTimer -= dt;
    if (state.pickupTimer <= 0) {
      state.pickupTimer = rand(9, 14);
      if (state.pickups.length < 2) spawnPickup();
    }
  }

  // ============================================================
  //  Update
  // ============================================================
  function hurt(dmg) {
    const p = state.player;
    if (p.invuln > 0) return false;
    p.hp = Math.max(0, p.hp - dmg);
    p.invuln = HIT_INVULN;
    p.sinceHit = 0;
    if (p.hp <= 0) gameOver();
    return true;
  }

  // Distance from point (px,py) to the infinite line through (ox,oy) at angle a.
  function distToLine(px, py, ox, oy, a) {
    const dx = px - ox, dy = py - oy;
    const ca = Math.cos(a), sa = Math.sin(a);
    // perpendicular component magnitude
    return Math.abs(dx * sa - dy * ca);
  }

  // Rotate a projectile's velocity toward `targetAng` by at most maxTurn·dt
  // radians, preserving its speed magnitude. This is what makes a bullet
  // "home" — and the cap is what makes it dodgeable.
  function steerToward(b, targetAng, maxTurn, dt) {
    const cur = Math.atan2(b.vy, b.vx);
    let diff = targetAng - cur;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    const step = maxTurn * dt;
    if (diff > step) diff = step;
    else if (diff < -step) diff = -step;
    const next = cur + diff;
    const spd = Math.hypot(b.vx, b.vy);
    b.vx = Math.cos(next) * spd;
    b.vy = Math.sin(next) * spd;
  }

  function update(dt) {
    const p = state.player;
    state.elapsed += dt;
    p.invuln = Math.max(0, p.invuln - dt);
    p.sinceHit += dt;

    // AI autopilot refreshes its held move at the policy's 30 Hz cadence.
    if (state.aiOn) AI.tick(dt);

    // Movement.
    const mv = moveVector();
    p.x = clamp(p.x + mv.x * PLAYER_SPEED * dt, PLAYER_R, state.W - PLAYER_R);
    p.y = clamp(p.y + mv.y * PLAYER_SPEED * dt, PLAYER_R, state.H - PLAYER_R);

    // Regen after staying safe.
    if (p.sinceHit >= REGEN_DELAY && p.hp < MAX_HP) {
      p.hp = Math.min(MAX_HP, p.hp + REGEN_RATE * dt);
    }

    runDirector(dt);

    // Projectiles.
    const margin = 40;
    const alive = [];
    for (const b of state.projectiles) {
      const cfg = TYPES[b.kind];
      // Homing bullets turn toward the player, capped by their max turn rate.
      if (b.kind === 'aimed') {
        steerToward(b, Math.atan2(p.y - b.y, p.x - b.x), cfg.turn, dt);
      }
      b.x += b.vx * dt; b.y += b.vy * dt;
      // Steering types can loiter on-screen; expire them so they can't orbit
      // forever (counts as dodged → awards score).
      if (b.life !== undefined) {
        b.life -= dt;
        if (b.life <= 0) {
          if (!b.hitOnce && !b.scored) addScore(cfg.score);
          continue;
        }
      }
      // Collision.
      if (!b.hitOnce) {
        const dx = b.x - p.x, dy = b.y - p.y;
        if (dx * dx + dy * dy <= (b.r + PLAYER_R) * (b.r + PLAYER_R)) {
          if (hurt(cfg.dmg)) { b.hitOnce = true; }   // counts as a hit, no score
        }
      }
      // Off-screen → dodged (if it never hit) → award score, drop it.
      if (b.x < -margin || b.x > state.W + margin ||
          b.y < -margin || b.y > state.H + margin) {
        if (!b.hitOnce && !b.scored) { addScore(cfg.score); }
        continue;   // remove
      }
      alive.push(b);
    }
    state.projectiles = alive;

    // Lasers.
    const laserAlive = [];
    for (const L of state.lasers) {
      if (L.phase === 'charge') {
        L.charge -= dt;
        if (L.charge <= 0) L.phase = 'fire';
      } else {
        L.fire -= dt;
        // Damage on contact with the beam. hurt()'s i-frames keep a single
        // sweep from draining the whole bar in one frame, while still letting
        // a player who lingers in the beam get tagged again after i-frames.
        if (distToLine(p.x, p.y, L.ox, L.oy, L.ang) <= L.halfWidth + PLAYER_R) {
          if (hurt(TYPES.laser.dmg)) L.hitOnce = true;
        }
        if (L.fire <= 0) {
          if (!L.hitOnce) addScore(TYPES.laser.score);
          continue;   // remove
        }
      }
      laserAlive.push(L);
    }
    state.lasers = laserAlive;

    // Bombs: tick the fuse, detonate, then linger briefly as a flash.
    const bombAlive = [];
    for (const bm of state.bombs) {
      if (bm.phase === 'fuse') {
        bm.fuse -= dt;
        if (bm.fuse <= 0) {
          bm.phase = 'flash';
          bm.flash = BOMB_FLASH;
          // Area damage to anyone caught in the blast radius at detonation.
          const dx = p.x - bm.x, dy = p.y - bm.y;
          if (dx * dx + dy * dy <= bm.blastR * bm.blastR) {
            if (hurt(TYPES.bomb.dmg)) bm.hitOnce = true;
          }
        }
      } else {
        bm.flash -= dt;
        if (bm.flash <= 0) {
          if (!bm.hitOnce) addScore(TYPES.bomb.score);
          continue;   // remove
        }
      }
      bombAlive.push(bm);
    }
    state.bombs = bombAlive;

    // Health packs: heal on contact (only if not already full, so a full-HP
    // player doesn't waste one), and fade out after their lifetime.
    const pickAlive = [];
    for (const k of state.pickups) {
      k.life -= dt;
      const dx = k.x - p.x, dy = k.y - p.y;
      const touching = dx * dx + dy * dy <= (PICKUP_R + PLAYER_R) * (PICKUP_R + PLAYER_R);
      if (touching && p.hp < MAX_HP) {
        p.hp = Math.min(MAX_HP, p.hp + HEAL_AMOUNT);
        continue;   // consumed
      }
      if (k.life <= 0) continue;   // expired
      pickAlive.push(k);
    }
    state.pickups = pickAlive;
  }

  function addScore(n) {
    state.score += n;
    state.dodged += 1;
  }

  // ============================================================
  //  Render
  // ============================================================
  function draw() {
    const W = state.W, H = state.H;
    // Letterbox: clear the WHOLE backing canvas (incl. the centered bands), then
    // clip the rest of drawing to the 450×600 play area so nothing spills out.
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();

    // Subtle grid backdrop.
    const isDark = document.documentElement.dataset.theme === 'dark';
    ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
    ctx.lineWidth = 1;
    const step = 40;
    ctx.beginPath();
    for (let x = step; x < W; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let y = step; y < H; y += step) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();

    // Lasers (under projectiles).
    for (const L of state.lasers) {
      const len = Math.hypot(W, H) * 2;
      const ca = Math.cos(L.ang), sa = Math.sin(L.ang);
      const x1 = L.ox - ca * len, y1 = L.oy - sa * len;
      const x2 = L.ox + ca * len, y2 = L.oy + sa * len;
      if (L.phase === 'charge') {
        // Thin pulsing telegraph line.
        const pulse = 0.35 + 0.35 * Math.abs(Math.sin(L.charge * 14));
        ctx.save();
        ctx.globalAlpha = pulse;
        ctx.strokeStyle = TYPES.laser.color;
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 8]);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.restore();
      } else {
        // Full beam with glow.
        ctx.save();
        ctx.strokeStyle = TYPES.laser.color;
        ctx.shadowColor = TYPES.laser.glow;
        ctx.shadowBlur = 18;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = L.halfWidth * 2;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.lineWidth = L.halfWidth;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.restore();
      }
    }

    // Health packs (under bullets — they're targets, not threats).
    for (const k of state.pickups) {
      const blink = k.life < 2 && (Math.floor(k.life * 8) % 2 === 0);
      ctx.save();
      ctx.globalAlpha = blink ? 0.35 : 1;
      ctx.shadowColor = HEAL_COLOR;
      ctx.shadowBlur = 12;
      ctx.fillStyle = HEAL_COLOR;
      // Rounded-square medkit so it reads differently from the green bullets.
      const s = PICKUP_R, x0 = k.x - s, y0 = k.y - s, d = s * 2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y0, d, d, 5);
      else ctx.rect(x0, y0, d, d);
      ctx.fill();
      // White cross.
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(k.x - s * 0.5, k.y); ctx.lineTo(k.x + s * 0.5, k.y);
      ctx.moveTo(k.x, k.y - s * 0.5); ctx.lineTo(k.x, k.y + s * 0.5);
      ctx.stroke();
      ctx.restore();
    }

    // Bomb fuse telegraph (danger zone + body). The explosion flash is drawn
    // later, on top of everything.
    for (const bm of state.bombs) {
      if (bm.phase !== 'fuse') continue;
      const prog = 1 - bm.fuse / BOMB_FUSE;          // 0 → 1 as it nears detonation
      ctx.save();
      // Pulsing dashed danger ring — pulse speeds up as the fuse runs down.
      ctx.globalAlpha = 0.3 + 0.3 * Math.abs(Math.sin(bm.fuse * (6 + prog * 16)));
      ctx.strokeStyle = TYPES.bomb.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 7]);
      ctx.beginPath(); ctx.arc(bm.x, bm.y, bm.blastR, 0, Math.PI * 2); ctx.stroke();
      // Filling pie wedge as a countdown.
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = TYPES.bomb.color;
      ctx.beginPath();
      ctx.moveTo(bm.x, bm.y);
      ctx.arc(bm.x, bm.y, bm.blastR, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2);
      ctx.closePath(); ctx.fill();
      // Bomb body, pulsing.
      ctx.globalAlpha = 1;
      ctx.shadowColor = TYPES.bomb.glow;
      ctx.shadowBlur = 10;
      ctx.fillStyle = TYPES.bomb.color;
      const pr = TYPES.bomb.r * (1 + 0.18 * Math.sin(bm.fuse * 16));
      ctx.beginPath(); ctx.arc(bm.x, bm.y, pr, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // Projectiles.
    for (const b of state.projectiles) {
      const cfg = TYPES[b.kind];
      ctx.save();
      ctx.shadowColor = cfg.glow;
      ctx.shadowBlur = b.kind === 'cannon' ? 14 : 8;
      ctx.fillStyle = cfg.color;
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
      ctx.fill();
      if (b.kind === 'cannon') {
        // Inner highlight to read as a heavy shell.
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.beginPath();
        ctx.arc(b.x - b.r * 0.3, b.y - b.r * 0.3, b.r * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Player — the chosen avatar (cosmetic), drawn larger than the hitbox,
    // with a small hitbox dot at the true collision center.
    const p = state.player;
    if (state.mode !== 'ready') {
      const blink = p.invuln > 0 && (Math.floor(p.invuln * 20) % 2 === 0);
      ctx.save();
      ctx.globalAlpha = blink ? 0.4 : 1;
      // Soft grounding shadow under the avatar.
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + AVATAR_R * 0.95, AVATAR_R * 0.7, AVATAR_R * 0.28, 0, 0, Math.PI * 2);
      ctx.fill();
      drawAvatar(ctx, avatarId, p.x, p.y, AVATAR_R);
      ctx.restore();
      // Hitbox indicator: a small hollow ring (not a filled dot) so the
      // avatar shows through the middle while the true collision center stays
      // clearly marked. Dark outer + white inner stroke reads on any backdrop.
      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.4, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.4, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
    }

    // Bomb explosion flash — drawn last so the blast reads on top of all.
    for (const bm of state.bombs) {
      if (bm.phase !== 'flash') continue;
      const a = Math.max(0, bm.flash / BOMB_FLASH);   // 1 → 0 fade
      ctx.save();
      ctx.globalAlpha = a;
      const grad = ctx.createRadialGradient(bm.x, bm.y, 0, bm.x, bm.y, bm.blastR);
      grad.addColorStop(0, 'rgba(255,255,255,0.95)');
      grad.addColorStop(0.45, TYPES.bomb.color);
      grad.addColorStop(1, 'rgba(249,115,22,0)');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(bm.x, bm.y, bm.blastR, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
    ctx.restore();   // end letterbox clip (paired with the save() at draw() start)
  }

  function drawHud() {
    const p = state.player;
    const pct = Math.max(0, p.hp) / MAX_HP * 100;
    els.hpFill.style.width = pct + '%';
    els.hpFill.classList.toggle('low', pct <= 50 && pct > 25);
    els.hpFill.classList.toggle('crit', pct <= 25);
    els.time.textContent = state.elapsed.toFixed(1) + 's';
    els.score.textContent = state.score;
  }

  // ============================================================
  //  Main loop
  // ============================================================
  function frame(ts) {
    state.raf = requestAnimationFrame(frame);
    if (state.mode !== 'playing') return;
    if (!state.lastTs) state.lastTs = ts;
    let dt = (ts - state.lastTs) / 1000;
    state.lastTs = ts;
    // Guard against huge dt after a stall (tab switch) — sub-step instead.
    dt = Math.min(dt, 0.05);
    update(dt);
    draw();
    drawHud();
  }

  // ============================================================
  //  Game flow
  // ============================================================
  function startGame() {
    resize();
    state.mode = 'playing';
    // Start dead-centre to match the SAC training env (env_dodge.py resets the
    // player to W*0.5, H*0.5); the old H*0.75 put the opening ~1s off-distribution.
    state.player = { x: state.W / 2, y: state.H * 0.5, hp: MAX_HP, sinceHit: 999, invuln: 0 };
    state.projectiles = [];
    state.lasers = [];
    state.bombs = [];
    state.pickups = [];
    state.score = 0;
    state.dodged = 0;
    state.aiEverUsed = state.aiOn;   // a round that starts under AI is already tainted
    els.aiHint.hidden = !state.aiEverUsed;

    state.elapsed = 0;
    state.spawnTimer = 0.6;
    state.ringTimer = 2.5;
    state.laserTimer = 5;
    state.bombTimer = 7;
    state.pickupTimer = 8;
    state.recentKinds = [];
    state.lastRing = null;
    state.lastTs = 0;
    keys.clear();
    AI.reset();
    hideOverlay();
    els.startBtn.textContent = t('restart');
    els.pauseBtn.textContent = t('pause');
    els.pauseBtn.disabled = false;
    els.stage.focus({ preventScroll: true });
    drawHud();
    if (!state.raf) state.raf = requestAnimationFrame(frame);
  }

  function pauseGame() {
    if (state.mode !== 'playing') return;
    state.mode = 'paused';
    els.pauseBtn.textContent = t('resume');
    renderOverlay();
    showOverlay();
  }

  function resumeGame() {
    if (state.mode !== 'paused') return;
    state.mode = 'playing';
    state.lastTs = 0;
    els.pauseBtn.textContent = t('pause');
    hideOverlay();
    els.stage.focus({ preventScroll: true });
  }

  function gameOver() {
    state.mode = 'gameover';
    if (state.score > best) {
      best = state.score;
      try { localStorage.setItem('cd_best', String(best)); } catch(_) {}
    }
    els.best.textContent = best;
    els.startBtn.textContent = t('start');
    els.pauseBtn.textContent = t('pause');
    els.pauseBtn.disabled = true;
    // AI-driven runs never post to the leaderboard (honor system).
    if (window.SkzLB && !state.aiEverUsed) window.SkzLB.submit('dodge', state.score);
    renderOverlay();
    showOverlay();
  }

  function showOverlay() { els.overlay.classList.add('show'); }
  function hideOverlay() { els.overlay.classList.remove('show'); }

  function renderOverlay() {
    const o = els;
    if (state.mode === 'ready') {
      o.overlayTitle.textContent = t('intro_title');
      o.overlaySub.textContent = t('intro_sub');
      o.overlaySub.hidden = false;
      o.overlayStats.hidden = true;
      o.overlayBtn.textContent = t('start');
    } else if (state.mode === 'paused') {
      o.overlayTitle.textContent = t('paused_title');
      o.overlaySub.textContent = t('paused_sub');
      o.overlaySub.hidden = false;
      o.overlayStats.hidden = true;
      o.overlayBtn.textContent = t('resume');
    } else if (state.mode === 'gameover') {
      o.overlayTitle.textContent = t('over_title');
      const isBest = state.score >= best && state.score > 0;
      o.overlaySub.textContent = isBest ? t('newbest') : '';
      o.overlaySub.hidden = !isBest;
      o.overlayStats.hidden = false;
      o.overlayStats.innerHTML =
        t('stat_score') + ': <b>' + state.score + '</b><br>' +
        t('stat_time') + ': ' + state.elapsed.toFixed(1) + 's · ' +
        t('stat_dodged') + ': ' + state.dodged +
        (state.aiEverUsed ? '<br><span class="ai-note">' + t('ai_note') + '</span>' : '');
      o.overlayBtn.textContent = t('restart');
    }
  }

  // ---------- button wiring ----------
  els.overlayBtn.addEventListener('click', () => {
    if (state.mode === 'paused') resumeGame();
    else startGame();
  });
  els.startBtn.addEventListener('click', startGame);
  els.pauseBtn.addEventListener('click', () => {
    if (state.mode === 'playing') pauseGame();
    else if (state.mode === 'paused') resumeGame();
  });

  // AI autopilot toggle: hand control to the SAC policy. Turning it on from a
  // menu auto-starts a round so you can just sit back and watch it play.
  function setAi(on) {
    if (on && !AI.load()) return;             // weights missing/bad — no-op
    state.aiOn = on;
    if (on) state.aiEverUsed = true;          // sticky: taints this round's score
    AI.reset();
    els.aiBtn.setAttribute('aria-pressed', String(on));
    els.aiBtn.classList.toggle('primary', on);
    els.aiBtn.innerHTML = withBrand(on ? t('ai_on') : t('ai_play'));
    els.aiHint.hidden = !state.aiEverUsed;   // sticky caption: AI use disqualifies this run
    if (on && state.mode !== 'playing') startGame();
    else if (state.mode === 'playing') els.stage.focus({ preventScroll: true });
  }
  // Handing over to the AI disqualifies the run from the leaderboard; instead of
  // a modal we surface a quiet caption under the buttons (see setAi / ai_hint).
  els.aiBtn.addEventListener('click', () => {
    setAi(!state.aiOn);
  });

  // Auto-pause when the tab/window loses focus mid-game.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.mode === 'playing') pauseGame();
  });
  window.addEventListener('blur', () => {
    if (state.mode === 'playing') pauseGame();
  });

  // ---------- boot ----------
  els.best.textContent = best;
  if (!AI.available) els.aiBtn.style.display = 'none';   // policy file absent
  applyLang();
  requestAnimationFrame(() => {
    resize();
    renderOverlay();
    draw();
    // Keep the render loop alive so overlays/joystick draw between rounds.
    state.raf = requestAnimationFrame(frame);
  });
})();
