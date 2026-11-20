(function (scope) {
  'use strict';

  var LANG_KEY = 'skz_lang';
  var THEME_KEY = 'skz_theme';
  var THEMES = ['auto', 'light', 'dark'];
  var CONTROL_LABELS = {
    zh: { lang: '切换语言', theme: '切换配色主题', auto: '自动', light: '明亮', dark: '暗黑' },
    en: { lang: 'Switch language', theme: 'Toggle theme', auto: 'Auto', light: 'Light', dark: 'Dark' },
  };
  var langListeners = [];
  var themeListeners = [];

  function getLang() {
    try {
      var l = localStorage.getItem(LANG_KEY);
      return l === 'en' ? 'en' : 'zh';
    } catch (_) {
      return 'zh';
    }
  }

  function getThemeMode() {
    try {
      var m = localStorage.getItem(THEME_KEY);
      return m === 'light' || m === 'dark' ? m : 'auto';
    } catch (_) {
      return 'auto';
    }
  }

  function resolveTheme(mode) {
    if (mode === 'light' || mode === 'dark') return mode;
    try {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } catch (_) {
      return 'light';
    }
  }

  function syncLangUI(lang) {
    var labels = CONTROL_LABELS[lang] || CONTROL_LABELS.zh;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    var langSeg = document.getElementById('lang_seg');
    if (langSeg) langSeg.setAttribute('aria-label', labels.lang);
    document.querySelectorAll('#lang_seg .seg-btn[data-lang]').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.lang === lang));
    });
    var themeSeg = document.getElementById('theme_seg');
    if (themeSeg) {
      themeSeg.setAttribute('aria-label', labels.theme);
      themeSeg.querySelectorAll('.seg-btn[data-theme]').forEach(function (b) {
        var title = labels[b.dataset.theme];
        if (title) {
          b.setAttribute('aria-label', title);
          b.setAttribute('title', title);
        }
      });
    }
  }

  function syncThemeUI(mode, resolved) {
    document.documentElement.dataset.theme = resolved;
    document.documentElement.dataset.themeMode = mode;
    document.querySelectorAll('#theme_seg .seg-btn[data-theme]').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.theme === mode));
    });
  }

  function setLang(lang) {
    var next = lang === 'en' ? 'en' : 'zh';
    try { localStorage.setItem(LANG_KEY, next); } catch (_) {}
    syncLangUI(next);
    langListeners.forEach(function (fn) { try { fn(next); } catch (_) {} });
    window.dispatchEvent(new CustomEvent('skz:lang', { detail: { lang: next } }));
  }

  function applyTheme(mode) {
    if (THEMES.indexOf(mode) < 0) mode = 'auto';
    var resolved = resolveTheme(mode);
    syncThemeUI(mode, resolved);
    themeListeners.forEach(function (fn) { try { fn(resolved, mode); } catch (_) {} });
    window.dispatchEvent(new CustomEvent('skz:theme', { detail: { mode: mode, theme: resolved } }));
  }

  function setThemeMode(mode) {
    if (THEMES.indexOf(mode) < 0) mode = 'auto';
    try {
      if (mode === 'auto') localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, mode);
    } catch (_) {}
    applyTheme(mode);
  }

  function bindControls() {
    syncLangUI(getLang());
    applyTheme(getThemeMode());
    document.querySelectorAll('#lang_seg .seg-btn[data-lang]').forEach(function (b) {
      if (b._skzBound) return;
      b._skzBound = true;
      b.addEventListener('click', function () {
        if (b.dataset.lang !== getLang()) setLang(b.dataset.lang);
      });
    });
    document.querySelectorAll('#theme_seg .seg-btn[data-theme]').forEach(function (b) {
      if (b._skzBound) return;
      b._skzBound = true;
      b.addEventListener('click', function () {
        setThemeMode(b.dataset.theme);
      });
    });
  }

  try {
    var mql = window.matchMedia('(prefers-color-scheme: dark)');
    var onSys = function () {
      if ((document.documentElement.dataset.themeMode || 'auto') === 'auto') applyTheme('auto');
    };
    if (mql.addEventListener) mql.addEventListener('change', onSys);
    else if (mql.addListener) mql.addListener(onSys);
  } catch (_) {}

  bindControls();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindControls);
  }

  scope.SkzPrefs = {
    getLang: getLang,
    setLang: setLang,
    getThemeMode: getThemeMode,
    setThemeMode: setThemeMode,
    resolveTheme: resolveTheme,
    onLang: function (fn) { langListeners.push(fn); },
    onTheme: function (fn) { themeListeners.push(fn); },
    bindControls: bindControls,
  };
})(window);
