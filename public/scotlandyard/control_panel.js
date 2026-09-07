import {ticketCountPresentation} from './action_view.js';

export const GAMEPLAY_FIELDS = [
  'num_pursuers',
  'max_rounds',
  'reveal_rounds',
  'ruleset',
  'evader_start_mode',
  'evader_start_locations',
  'pursuer_start_mode',
  'pursuer_start_locations',
  'pursuer_history_length',
];

export const FIELD_LABELS = {
  num_pursuers: 'Pursuer 数量',
  max_rounds: '最大回合数',
  reveal_rounds: '暴露回合',
  ruleset: '规则集',
  environment_identity: '规则内容',
  evader_start_mode: 'Evader 起点规则',
  evader_start_locations: 'Evader 起点',
  pursuer_start_mode: 'Pursuer 起点规则',
  pursuer_start_locations: 'Pursuer 起点',
  pursuer_history_length: 'Pursuer 历史长度',
};

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function gameMismatches(left, right) {
  if (!left || !right) return [];
  return [...GAMEPLAY_FIELDS, 'environment_identity'].filter(
    field => !valuesEqual(left[field], right[field]),
  );
}

export function strategySupportsRuleset(strategy, ruleset) {
  if (['random', 'heuristic', 'human'].includes(strategy.kind)) return true;
  return strategy.game?.ruleset === ruleset;
}

export function supportText(strategy) {
  if (!strategy) return '玩家控制';
  const support = [...(strategy.support || [])]
    .sort((left, right) => Number(right.probability) - Number(left.probability));
  const shown = support.slice(0, 3).map(item =>
    `${item.policy_id} ${(100 * item.probability).toFixed(1)}%`);
  if (support.length > shown.length) shown.push(`另 ${support.length - shown.length} 个策略`);
  return shown.join(' · ') || strategy.label;
}

export function actionModeLabel(mode) {
  if (String(mode || '').startsWith('az_mcts_')) {
    return `AZ MCTS · ${String(mode).slice('az_mcts_'.length)} 次模拟`;
  }
  return {
    record: '产物记录的默认模式',
    sample: '按分布采样',
    deterministic_argmax: '确定性 argmax',
    random: '随机动作',
    heuristic: '启发式决策',
    human: '玩家选择',
    az_network_greedy: 'AZ 纯网络（greedy）',
  }[mode] || mode || '未设置';
}

export function controllerKindLabel(source, policyId = '') {
  if (!source) return policyId || '未知';
  if (source.kind === 'random') return '内置随机控制器';
  if (source.kind === 'human') return '玩家控制';
  if (source.kind === 'heuristic') return `内置启发式 · ${source.name}`;
  if (source.kind === 'ppo') return `PPO · ${source.run_id}/${source.checkpoint}`;
  if (source.kind === 'psro') return `PSRO · ${source.run_id}/${source.strategy_id}`;
  if (source.kind === 'az_selfplay') return `AZ Selfplay · ${source.run_id}/gen_${String(source.generation).padStart(6, '0')}`;
  if (source.kind === 'az') return `AZ · ${source.run_id}/gen_${String(source.generation).padStart(6, '0')}`;
  return policyId || source.kind;
}

function strategyGroup(strategy, role) {
  if (strategy.kind === 'random' || strategy.kind === 'heuristic') return '内置控制器';
  if (strategy.kind === 'az_model') return `${strategy.route === 'az_selfplay' ? 'AZ Selfplay' : 'AZ'} · ${strategy.run_id || '未命名运行'}`;
  if (strategy.kind === 'ppo_checkpoint') return `PPO · ${strategy.run_id || '未命名运行'}`;
  if (strategy.kind.startsWith('psro_')) return `PSRO · ${strategy.run_id || '未命名运行'}`;
  return role === 'pursuer' ? 'Pursuer 策略' : 'Evader 策略';
}

export function createControlPanel({$, element, addKeyValue, session, getStatus}) {
  const strategySelections = {pursuer: null, evader: 'human'};

  function option(label, value) {
    const node = document.createElement('option');
    node.textContent = label;
    node.value = value;
    return node;
  }

  function populateStrategySelect(select, role, strategies, includeHuman = false, query = '') {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const matches = strategy => !normalizedQuery || [
      strategy.label,
      strategy.run_id,
      strategy.kind,
      strategy.source,
    ].some(value => String(value || '').toLocaleLowerCase().includes(normalizedQuery));
    const filtered = strategies.filter(strategy => strategySupportsRuleset(strategy, $('ruleset-select').value) && matches(strategy));
    const groups = new Map();
    if (includeHuman && (!normalizedQuery || '玩家控制 Evader human'.toLocaleLowerCase().includes(normalizedQuery))) {
      groups.set('玩家控制', [option('玩家控制 Evader', 'human')]);
    }
    for (const strategy of filtered) {
      const name = strategyGroup(strategy, role);
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(option(strategy.label, strategy.id));
    }
    const nodes = [];
    for (const [label, options] of groups) {
      const group = document.createElement('optgroup');
      group.label = label;
      group.append(...options);
      nodes.push(group);
    }
    if (!nodes.length) {
      const empty = option('没有匹配的策略', '');
      empty.disabled = true;
      empty.selected = true;
      select.replaceChildren(empty);
      select.disabled = true;
      return;
    }
    select.replaceChildren(...nodes);
    const remembered = strategySelections[role];
    const rememberedOption = remembered
      ? [...select.options].find(item => item.value === remembered)
      : null;
    if (rememberedOption) {
      select.value = remembered;
    } else if (normalizedQuery && remembered) {
      const prompt = option('请选择筛选结果…', '');
      prompt.disabled = true;
      prompt.selected = true;
      select.prepend(prompt);
    } else {
      strategySelections[role] = select.value;
    }
    select.disabled = false;
  }

  function strategyById(role, id) {
    return (session.strategyData?.[role] || []).find(item => item.id === id) || null;
  }

  function selectedPursuer() {
    return strategyById('pursuer', $('pursuer-select').value);
  }

  function selectedEvader() {
    return $('evader-select').value === 'human'
      ? null
      : strategyById('evader', $('evader-select').value);
  }

  function selectedMismatches() {
    return gameMismatches(selectedPursuer()?.game, selectedEvader()?.game);
  }

  function selectedRules() {
    const pursuer = selectedPursuer();
    const evader = selectedEvader();
    if (pursuer?.game) return {game: pursuer.game, source: '由 Pursuer 策略决定'};
    if (evader?.game) return {game: evader.game, source: '由 Evader 策略决定'};
    return {game: session.strategyData?.rulesets?.[$('ruleset-select').value] || session.strategyData?.default_game || null, source: '所选规则'};
  }

  function usesAZController() {
    return selectedPursuer()?.kind === 'az_model' || selectedEvader()?.kind === 'az_model';
  }

  function readAZSimulations() {
    const raw = $('az-simulations').value.trim();
    const value = Number(raw);
    if (!raw || !Number.isSafeInteger(value) || value < 0 || value > 100000) {
      throw new Error('AZ 模拟次数必须是 0 到 100000 之间的整数。');
    }
    return value;
  }

  function azModeSummary() {
    try {
      const simulations = readAZSimulations();
      const publicSearch = [selectedPursuer(), selectedEvader()].some(choice => choice?.route === 'az_selfplay');
      return simulations === 0 ? '纯网络（greedy）' : publicSearch
        ? `PUCT · 请求 ${simulations} 次，Selfplay 按公开候选位置数向上取整`
        : `MCTS · ${simulations} 次模拟`;
    } catch (_) {
      return '模拟次数无效';
    }
  }

  function renderRulesPreview() {
    const {game, source} = selectedRules();
    $('rules-source').textContent = game ? source : '规则不可用';
    const container = $('rules-preview');
    container.replaceChildren();
    if (!game) {
      container.append(element('p', 'empty', '无法读取规则。'));
      return;
    }
    const reveals = (game.reveal_rounds || []).join('、') || '无';
    addKeyValue(container, 'rule-item', '回合', `${game.max_rounds}`);
    addKeyValue(container, 'rule-item', 'Pursuer', `${game.num_pursuers} 名`);
    addKeyValue(container, 'rule-item', '暴露回合', reveals);
    addKeyValue(container, 'rule-item', 'Pursuer 票（Taxi／Bus／Underground）',
      game.resolved_ruleset.pursuer_initial_tickets.map(value => ticketCountPresentation(value, value)).join(' / '));
  }

  function renderStrategySummary() {
    if (!session.strategyData) return;
    const pursuer = selectedPursuer();
    const evader = selectedEvader();
    const pursuerSummary = $('pursuer-select').value ? supportText(pursuer) : '未选择';
    const evaderSummary = $('evader-select').value ? supportText(evader) : '未选择';
    const mode = pursuer?.kind === 'random'
      ? '随机动作（固定）'
      : pursuer?.kind === 'az_model'
        ? azModeSummary()
        : actionModeLabel($('action-mode-select').value);
    const lines = [
      `Pursuer：${pursuerSummary}`,
      `Evader：${evaderSummary}`,
      `Pursuer 推理：${mode}`,
    ];
    if (usesAZController()) lines.push(`AZ 控制器设置：${azModeSummary()}`);
    $('strategy-summary').textContent = lines.join('\n');
  }

  function refreshStartAvailability() {
    let invalidAZ = false;
    if (usesAZController()) {
      try { readAZSimulations(); } catch (_) { invalidAZ = true; }
    }
    const {actionInFlight, busyKind} = getStatus();
    const invalid = !session.catalogReady || actionInFlight || selectedMismatches().length > 0
      || !$('pursuer-select').value || !$('evader-select').value || invalidAZ;
    $('start').disabled = invalid;
    if (busyKind === 'start') $('start-label').textContent = '正在加载控制器…';
    else if (!session.catalogReady) $('start-label').textContent = '正在读取策略…';
    else $('start-label').textContent = session.gameId ? '以这些设置开始新对局' : '加载并开始';
  }

  function renderCompatibility() {
    const pursuer = selectedPursuer();
    for (const optionNode of $('evader-select').querySelectorAll('option')) {
      if (optionNode.value === 'human') continue;
      const evader = strategyById('evader', optionNode.value);
      const mismatches = gameMismatches(pursuer?.game, evader?.game);
      optionNode.disabled = mismatches.length > 0;
      optionNode.title = mismatches.length
        ? `规则不兼容：${mismatches.map(field => FIELD_LABELS[field] || field).join('、')}`
        : '';
    }
    const mismatches = selectedMismatches();
    const notice = $('compatibility');
    if (mismatches.length) {
      notice.textContent = `当前组合无法开始：${mismatches.map(field => FIELD_LABELS[field] || field).join('、')} 不一致。`;
      notice.hidden = false;
    } else {
      notice.hidden = true;
      notice.textContent = '';
    }
    const fixedPursuerMode = true;
    $('action-mode-field').hidden = fixedPursuerMode;
    $('action-mode-select').disabled = fixedPursuerMode;
    $('az-simulations-field').hidden = !usesAZController();
    $('az-simulations').disabled = !usesAZController();
    renderRulesPreview();
    renderStrategySummary();
    refreshStartAvailability();
  }

  function filterStrategies(role) {
    populateStrategySelect(
      $(`${role}-select`),
      role,
      session.strategyData?.[role] || [],
      role === 'evader',
      $(`${role}-search`).value,
    );
    renderCompatibility();
  }

  function renderCatalogNotice() {
    const notice = $('catalog-notice');
    const summary = session.strategyData.summary || {};
    const learned = Number(summary.learned_evader_count || 0)
      + Number(summary.learned_pursuer_count || 0);
    const errors = session.strategyData.errors || [];
    if (!learned) {
      notice.textContent = '未发现 PPO/PSRO/AZ 模型产物，当前仅提供内置 Random、Greedy 与 Territory 控制器。';
      notice.className = 'notice info';
      notice.hidden = false;
    } else {
      notice.textContent = '你控制 Evader，对手在浏览器本地运行。选择权重后即可开始。';
      notice.className = 'notice success';
      notice.hidden = false;
    }
    const details = $('catalog-details');
    const list = $('catalog-errors');
    list.replaceChildren(...errors.map(item =>
      element('li', '', `${item.source}/${item.run_id}：${item.error}`)));
    details.hidden = errors.length === 0;
    if (errors.length) {
      notice.className = 'notice';
      notice.textContent += ` 另有 ${errors.length} 个运行读取失败，详见“策略目录问题”。`;
    }
  }

  function rememberSelection(role, value) {
    strategySelections[role] = value;
  }

  return {
    azModeSummary,
    filterStrategies,
    populateStrategySelect,
    readAZSimulations,
    refreshStartAvailability,
    rememberSelection,
    renderCatalogNotice,
    renderCompatibility,
    renderStrategySummary,
    selectedEvader,
    selectedMismatches,
    selectedPursuer,
    strategyById,
    usesAZController,
  };
}
