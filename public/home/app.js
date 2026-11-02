    // ---------- i18n (self-contained for the landing page) ----------
    var STRINGS = {
      zh: {
        title: 'SkyZero · Web',
        hero_title: '选择一个游戏',
        gomoku_name: '五子棋',
        gomoku_desc: '和 SkyZero 对弈',
        hex_name: 'Hex 六角棋',
        hex_desc: '连接两边，挑战 AI',
        xiangqi_name: '中国象棋',
        xiangqi_desc: '楚河汉界，挑战 AI',
        tree_name: '搜索树可视化',
        tree_desc: 'AlphaZero MCTS 逐步展开',
        g2048_name: '2048',
        g2048_desc: '经典数字合并益智游戏',
        poker_name: '德州扑克',
        poker_desc: '与 DeepSeek 对弈 · PokerSkill 策略辅助',
        dodge_name: '躲避',
        dodge_desc: '操控人物躲避四面弹幕',
        aria_lang: '切换语言', aria_theme: '切换配色主题',
        th_auto: '自动', th_light: '明亮', th_dark: '暗黑',
      },
      en: {
        title: 'SkyZero · Web',
        hero_title: 'Choose a game',
        gomoku_name: 'Gomoku',
        gomoku_desc: 'Play against SkyZero',
        hex_name: 'Hex',
        hex_desc: 'Connect two edges against AI',
        xiangqi_name: 'Xiangqi',
        xiangqi_desc: 'Chinese chess against AI',
        tree_name: 'Search Tree',
        tree_desc: 'Watch AlphaZero MCTS expand',
        g2048_name: '2048',
        g2048_desc: 'The classic tile-merging puzzle',
        poker_name: 'Texas Hold’em',
        poker_desc: 'Play DeepSeek with PokerSkill strategy',
        dodge_name: 'Dodge',
        dodge_desc: 'Steer a character through bullet hell',
        aria_lang: 'Switch language', aria_theme: 'Toggle theme',
        th_auto: 'Auto', th_light: 'Light', th_dark: 'Dark',
      },
    };
    function getLang() {
      try { var l = localStorage.getItem('skz_lang'); return (l === 'zh' || l === 'en') ? l : 'zh'; }
      catch (e) { return 'zh'; }
    }
    function applyLang(lang) {
      var s = STRINGS[lang] || STRINGS.zh;
      document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
      document.title = s.title;
      document.getElementById('hero_title').textContent = s.hero_title;
      document.getElementById('card_gomoku_name').textContent = s.gomoku_name;
      document.getElementById('card_gomoku_desc').textContent = s.gomoku_desc;
      document.getElementById('card_hex_name').textContent = s.hex_name;
      document.getElementById('card_hex_desc').textContent = s.hex_desc;
      document.getElementById('card_xiangqi_name').textContent = s.xiangqi_name;
      document.getElementById('card_xiangqi_desc').textContent = s.xiangqi_desc;
      document.getElementById('card_tree_name').textContent = s.tree_name;
      document.getElementById('card_tree_desc').textContent = s.tree_desc;
      document.getElementById('card_2048_name').textContent = s.g2048_name;
      document.getElementById('card_2048_desc').textContent = s.g2048_desc;
      document.getElementById('card_dodge_name').textContent = s.dodge_name;
      document.getElementById('card_dodge_desc').textContent = s.dodge_desc;
      document.getElementById('card_poker_name').textContent = s.poker_name;
      document.getElementById('card_poker_desc').textContent = s.poker_desc;
      var langSeg = document.getElementById('lang_seg');
      langSeg.setAttribute('aria-label', s.aria_lang);
      langSeg.querySelectorAll('.seg-btn[data-lang]').forEach(function(b){
        b.setAttribute('aria-pressed', b.dataset.lang === lang ? 'true' : 'false');
      });
      var themeSeg = document.getElementById('theme_seg');
      themeSeg.setAttribute('aria-label', s.aria_theme);
      var titles = { auto: s.th_auto, light: s.th_light, dark: s.th_dark };
      themeSeg.querySelectorAll('.seg-btn[data-theme]').forEach(function(b){
        b.setAttribute('aria-label', titles[b.dataset.theme]);
        b.setAttribute('title', titles[b.dataset.theme]);
      });
    }

    // ---------- Theme ----------
    var THEME_MODES = ['auto', 'light', 'dark'];
    function resolveTheme(mode) {
      if (mode === 'light' || mode === 'dark') return mode;
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    function applyTheme(mode) {
      if (THEME_MODES.indexOf(mode) < 0) mode = 'auto';
      document.documentElement.dataset.theme = resolveTheme(mode);
      document.documentElement.dataset.themeMode = mode;
      document.querySelectorAll('#theme_seg .seg-btn[data-theme]').forEach(function(b){
        b.setAttribute('aria-pressed', b.dataset.theme === mode ? 'true' : 'false');
      });
    }

    document.addEventListener('DOMContentLoaded', function () {
      applyLang(getLang());
      applyTheme(document.documentElement.dataset.themeMode || 'auto');

      document.querySelectorAll('#lang_seg .seg-btn[data-lang]').forEach(function (b) {
        b.addEventListener('click', function () {
          var lang = b.dataset.lang;
          try { localStorage.setItem('skz_lang', lang); } catch (e) {}
          applyLang(lang);
        });
      });
      document.querySelectorAll('#theme_seg .seg-btn[data-theme]').forEach(function (b) {
        b.addEventListener('click', function () {
          var next = b.dataset.theme;
          if (THEME_MODES.indexOf(next) < 0) return;
          try {
            if (next === 'auto') localStorage.removeItem('skz_theme');
            else localStorage.setItem('skz_theme', next);
          } catch (e) {}
          applyTheme(next);
        });
      });

      try {
        var mql = window.matchMedia('(prefers-color-scheme: dark)');
        var onSys = function () {
          if ((document.documentElement.dataset.themeMode || 'auto') !== 'auto') return;
          applyTheme('auto');
        };
        if (mql.addEventListener) mql.addEventListener('change', onSys);
        else if (mql.addListener) mql.addListener(onSys);
      } catch (e) {}
    });
