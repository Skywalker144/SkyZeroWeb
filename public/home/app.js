    var STRINGS = {
      zh: {
        title: 'SkyZero · Web',
        hero_title: '选择一个游戏',
        gomoku_name: '五子棋', go_name: '围棋', go_desc: '黑白纵横，挑战 AI',
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
        scotlandyard_name: 'ScotlandYard',
        scotlandyard_desc: '扮演 Evader，挑战 AI 追捕队',
      },
      en: {
        title: 'SkyZero · Web',
        hero_title: 'Choose a game',
        gomoku_name: 'Gomoku', go_name: 'Go', go_desc: 'Black and white. Play the AI.',
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
        scotlandyard_name: 'ScotlandYard',
        scotlandyard_desc: 'Play Evader against AI pursuers',
      },
    };
    function applyLang(lang) {
      var s = STRINGS[lang] || STRINGS.zh;
      document.title = s.title;
      document.getElementById('hero_title').textContent = s.hero_title;
      document.getElementById('card_gomoku_name').textContent = s.gomoku_name;
      document.getElementById('card_gomoku_desc').textContent = s.gomoku_desc;
      document.getElementById('card_go_name').textContent = s.go_name;
      document.getElementById('card_go_desc').textContent = s.go_desc;
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
      document.getElementById('card_scotlandyard_name').textContent = s.scotlandyard_name;
      document.getElementById('card_scotlandyard_desc').textContent = s.scotlandyard_desc;
      document.getElementById('card_poker_name').textContent = s.poker_name;
      document.getElementById('card_poker_desc').textContent = s.poker_desc;
    }

    document.addEventListener('DOMContentLoaded', function () {
      if (window.SkzPrefs) {
        applyLang(window.SkzPrefs.getLang());
        window.SkzPrefs.onLang(applyLang);
      }
    });
