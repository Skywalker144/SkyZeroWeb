import {requestJson as api, setModelProgressListener} from './local-api.js?v=20260907';
import {legalActionRows, ticketCountPresentation, undoAvailable} from './action_view.js';
import {actionModeLabel} from './control_panel.js';
import {gameCreationRequest, gameSession as session} from './game_session.js';
import {createMapController} from './map_controller.js';
import {matchPresentation, winnerPresentation} from './match_view.js';
import {buildTurnTrack} from './timeline_view.js';

const $ = id => document.getElementById(id);

const TICKET_LABELS = {
  taxi: '出租车',
  bus: '公交',
  underground: '地铁',
  black: '黑票',
  forced_pass: '停留',
};

const TICKET_ORDER = ['taxi', 'bus', 'underground', 'black'];
let actionInFlight = false;
let autoplayRunning = false;
let autoplayTimer = null;
let busyKind = null;
let busyVisual = null;
const ANIMATION_STEP_MS = 100;

function element(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== '') node.textContent = text;
  return node;
}

function addKeyValue(container, className, label, value) {
  const item = element('div', className);
  item.append(element('small', '', label), element('b', '', String(value)));
  container.appendChild(item);
  return item;
}

function friendlyError(error) {
  const raw = String(error?.message || error || '未知错误');
  if (error?.code === 'game_not_found') return '当前对局已经失效。请重新载入页面后开始新对局。';
  if (raw.includes('Failed to fetch')) return '资源下载失败，请检查网络后重试。';
  if (raw.includes('seed must be a non-negative safe integer')) return 'seed 必须是 0 到 9007199254740991 之间的整数。';
  if (raw.includes('selected controllers use incompatible game rules')) {
    return `双方策略的游戏规则不兼容：${raw.split(':').slice(1).join(':').trim()}`;
  }
  if (raw.includes('game is already over')) return '本局已经结束，请重开或设置新对局。';
  if (raw.includes('no move to undo')) return '当前没有可以撤回的回合。';
  return raw;
}

function showError(error) {
  stopAutoplay();
  const node = $('error');
  const message = friendlyError(error);
  $('error-message').textContent = message;
  $('error-recover').hidden = error?.code !== 'game_not_found';
  node.hidden = false;
  $('status-live').textContent = message;
  if (error?.code === 'game_not_found') {
    session.gameId = null;
    session.rememberGameId(null);
    refreshActionAvailability();
  }
}

function clearError() {
  $('error').hidden = true;
  $('error-message').textContent = '';
  $('error-recover').hidden = true;
}

function announce(message) {
  $('status-live').textContent = '';
  window.setTimeout(() => { $('status-live').textContent = message; }, 0);
}

function backendLabel(value) {
  return {
    cpp: 'C++ / LibTorch',
    python: 'Python / TorchScript（调试）',
    python_torchscript: 'Python / TorchScript（直推）',
  }[value] || value || '未记录';
}

function deviceLabel(value) {
  const normalized = String(value || '').toLowerCase();
  if (normalized === 'cpu') return 'CPU';
  if (normalized === 'cuda') return 'CUDA';
  if (normalized.startsWith('cuda:')) {
    return `CUDA · GPU ${normalized.slice(5)}`;
  }
  return value || '自动选择';
}

function formatSeconds(ms) { return `${(ms / 1000).toFixed(1)} 秒`; }
function startBusyVisual(kind) {
  if (kind !== 'pursuer-turn') return;
  if (busyVisual) finishBusyVisual(false);
  const visual = {startedAt:performance.now(), kind};
  busyVisual = visual;
  $('search-title').textContent = kind === 'pursuer-turn' ? 'Pursuer 正在行动' : '正在加载对手';
  $('search-detail').textContent = kind === 'pursuer-turn' ? '纯网络推理' : '首次使用需要下载权重，请稍候';
  visual.interval = setInterval(() => {
    $('search-elapsed').textContent = `已用时 ${formatSeconds(performance.now()-visual.startedAt)}`;
  }, 100);
  $('search-overlay').hidden = false;
}
function finishBusyVisual(succeeded = true) {
  if (!busyVisual) return;
  clearInterval(busyVisual.interval);
  const elapsed = performance.now()-busyVisual.startedAt;
  busyVisual = null;
  $('search-overlay').hidden = true;
  if (succeeded) {
    $('search-feedback').textContent = `本地推理就绪 · ${formatSeconds(elapsed)}`;
    $('search-feedback').hidden = false;
  }
}

function generateSeed() {
  const values = new Uint32Array(1);
  if (globalThis.crypto?.getRandomValues) {
    crypto.getRandomValues(values);
    return String(values[0]);
  }
  return String(Date.now());
}

let nextSeed = Number(generateSeed());
function setNewSeed() { nextSeed = Number(generateSeed()); }

function syncSiteTheme() {
  let mode = 'auto';
  try { mode = localStorage.getItem('skz_theme') || 'auto'; } catch (_) {}
  if (!['auto', 'light', 'dark'].includes(mode)) mode = 'auto';
  const systemDark = matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.themeMode = mode;
  document.documentElement.dataset.theme = mode === 'auto'
    ? (systemDark ? 'dark' : 'light')
    : mode;
}

function setMatchLayout(active) {
  $('app').classList.toggle('has-game', active);
}

function refreshStartAvailability() {
  $('quick-pursuer').disabled = actionInFlight || !session.catalogReady;
}

function updateModelProgress({model, stage, loaded = 0, total = 1}) {
  $('model-loading').hidden = false;
  $('loading-retry').hidden = true;
  const name = model.id.startsWith('az-') ? 'AZ' : 'PPO GNN';
  const percent = Math.min(100, Math.floor(loaded / total * 100));
  $('model-progress').value = percent;
  $('loading-title').textContent = stage === 'download' ? `正在下载 ${name} 权重` : `正在加载 ${name}`;
  $('loading-detail').textContent = stage === 'download'
    ? `${percent}% · ${(loaded / 1048576).toFixed(2)} / ${(total / 1048576).toFixed(2)} MB`
    : stage === 'verify' ? '下载完成，正在校验…' : '下载完成，正在准备对局…';
}

const mapController = createMapController({
  $,
  element,
  session,
  ticketLabel,
  ticketOrder: TICKET_ORDER,
  announce,
  submitMove,
  selectNode,
  clearSelectedDestination,
  previewDestination,
  refreshActionAvailability,
  animationStepDuration: () => ANIMATION_STEP_MS,
});

const {
  animateTransition,
  cancelAnimation,
  fitBoardToArea,
  onMapClick,
  renderMap,
} = mapController;
function ticketLabel(ticket) {
  return TICKET_LABELS[ticket] || ticket;
}

function ticketCount(value, ticket, role) {
  const initial = session.gameMetadata.game_config.resolved_ruleset[`${role}_initial_tickets`];
  return ticketCountPresentation(value, initial[TICKET_ORDER.indexOf(ticket)]);
}

function winnerLabel(winner) {
  return winner === 'evader' ? 'Evader' : 'Pursuer';
}

function terminalReasonLabel(reason) {
  return {
    capture: 'Pursuer 完成抓捕',
    max_rounds: 'Evader 存活至回合上限',
    evader_stuck: 'Evader 无合法动作',
  }[reason] || reason || '对局结束';
}

function renderMatchCard() {
  const card = $('match-card');
  if (!session.state || !session.gameMetadata) {
    card.hidden = true;
    return;
  }
  const presentation = matchPresentation(session.state, session.gameMetadata);
  card.hidden = false;
  $('match-pursuer').textContent = presentation.pursuer;
  $('match-pursuer').title = presentation.pursuer;
  $('match-evader').textContent = presentation.evader;
  $('match-evader').title = presentation.evader;
}

function renderGameStatus(completedStepsOverride = null) {
  const progress = buildTurnTrack(session.state, completedStepsOverride);
  const track = $('round-track');
  const count = progress.totalSteps / 2;
  if (track.children.length !== count) {
    track.replaceChildren(...Array.from({length: count}, () => {
      const pair = element('span', 'round-pair');
      pair.append(element('span', 'round-fill'), element('span', 'round-head'),
        element('span', 'turn-marker'), element('span', 'turn-marker'));
      return pair;
    }));
  }
  track.setAttribute('role', 'list');
  track.setAttribute('aria-label', `对局行动进度，已完成 ${progress.completedSteps}/${progress.totalSteps} 步`);
  [...track.children].forEach((pair, index) => {
    const steps = progress.steps.slice(index * 2, index * 2 + 2);
    const reached = progress.currentStep;
    const extent = Math.max(0, Math.min(2, reached - index * 2));
    pair.dataset.extent = String(extent);
    pair.classList.toggle('complete', steps.every(step => step.played));
    pair.classList.toggle('active', !progress.terminal && steps.some(step => step.current));
    pair.setAttribute('role', 'listitem');
    pair.setAttribute('aria-label', steps.map(step => step.label + (step.exposedAt !== undefined
      ? ` · 暴露节点 ${step.exposedAt}` : step.scheduled ? ' · 计划暴露' : '')).join('；'));
    pair.title = pair.getAttribute('aria-label');
    if (pair.classList.contains('active')) pair.setAttribute('aria-current', 'step');
    else pair.removeAttribute('aria-current');
    steps.forEach((step, offset) => {
      const dot = pair.children[offset + 2];
      dot.classList.toggle('scheduled', step.scheduled);
      dot.classList.toggle('exposed', step.exposedAt !== undefined);
      dot.classList.toggle('reached', offset < extent);
    });
  });
  layoutRoundTrack();
}

function layoutRoundTrack() {
  const track = $('round-track');
  const columns = Math.min(
    Math.max(1, track.children.length),
    Math.max(1, Math.floor((track.clientWidth - 2) / 52)),
  );
  track.style.setProperty('--round-prefix-columns', String(columns - 1));
  track.classList.toggle('single-column', columns === 1);
  [...track.children].forEach((pair, index, pairs) => {
    pair.classList.toggle('row-end', (index + 1) % columns === 0 || index === pairs.length - 1);
  });
}

async function waitForTurnTrackAnimation() {
  const animations = $('round-track').getAnimations({subtree: true})
    .filter(animation => animation.playState !== 'finished');
  await Promise.allSettled(animations.map(animation => animation.finished));
}

function renderTicketBar() {
  const bar = $('ticket-bar');
  bar.replaceChildren();
  if (!session.state?.evader_tickets) return;
  for (const ticket of TICKET_ORDER) {
    const chip = element('span', 'ticket-chip');
    chip.dataset.ticket = ticket;
    chip.append(
      document.createTextNode(ticketLabel(ticket)),
      element('b', '', ticketCount(session.state.evader_tickets[ticket], ticket, 'evader')),
    );
    bar.appendChild(chip);
  }
}

function renderActionDock() {
  const list = $('action-list');
  list.replaceChildren();
  list.classList.remove('has-actions');
  list.removeAttribute('role');
  list.removeAttribute('aria-label');
  renderTicketBar();
  if (!session.state) {
    $('actions-title').textContent = '当前操作';
    $('actions-hint').textContent = '开局后显示合法动作';
    list.append(element('span', 'empty', '选择对手并开始逃脱。'));
    return;
  }
  if (session.state.terminal) {
    $('actions-title').textContent = 'Evader 合法动作';
    $('actions-hint').textContent = '对局已经结束';
    list.append(element('span', 'empty', '本局已结束，没有后续动作。'));
    return;
  }
  if (session.state.evader_mode === 'human' && session.state.phase === 'pursuer') {
    const calculating = busyKind === 'pursuer-turn';
    $('actions-title').textContent = calculating ? 'Pursuer 行动' : 'Evader 移动';
    $('actions-hint').textContent = calculating ? 'Evader 移动已经完成' : '正在播放玩家动作';
    list.append(element('span', 'empty', calculating
      ? '正在计算 Pursuer 的行动。'
      : '移动动画结束后开始计算 Pursuer 行动。'));
    return;
  }
  if (session.state.evader_mode !== 'human') {
    $('actions-title').textContent = '策略推进';
    $('actions-hint').textContent = `${session.state.evader_policy_id} · ${actionModeLabel(session.state.evader_action_mode)}`;
    list.append(element('span', 'empty', '使用“执行本轮”单步运行，或开启自动播放。'));
    return;
  }

  $('actions-title').textContent = 'Evader 合法动作';
  $('actions-hint').textContent = '选择目的地与交通方式';
  const rows = legalActionRows(session.state.legal_evader_moves, TICKET_ORDER);
  if (!rows.length) {
    list.append(element('span', 'empty', '没有合法动作。'));
    return;
  }
  list.classList.add('has-actions');
  list.setAttribute('role', 'group');
  list.setAttribute('aria-label', 'Evader 合法动作');
  for (const {destination, moves} of rows) {
    const group = element('div', `destination-group${mapController.selectedDestination === destination ? ' selected' : ''}`);
    group.dataset.destination = String(destination);
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', `目的地编号 ${destination}`);
    group.addEventListener('mouseenter', () => previewDestination(destination, true));
    group.addEventListener('mouseleave', () => previewDestination(destination, false));
    group.addEventListener('focusin', () => previewDestination(destination, true));
    group.addEventListener('focusout', event => {
      if (!group.contains(event.relatedTarget)) previewDestination(destination, false);
    });
    const destinationLabel = element('span', 'destination-label', String(destination));
    const transportCell = element('div', 'destination-moves');
    for (const move of moves) {
      const button = element('button', 'move-btn', ticketLabel(move.ticket));
      button.type = 'button';
      button.dataset.ticket = move.ticket;
      button.setAttribute('aria-label', `使用${ticketLabel(move.ticket)}前往节点 ${destination}`);
      button.addEventListener('click', () => submitMove(move));
      transportCell.appendChild(button);
    }
    group.append(destinationLabel, transportCell);
    list.appendChild(group);
  }
}

function previewDestination(destination, active) {
  const key = String(destination);
  for (const marker of document.querySelectorAll('.legal-marker')) {
    if (marker.dataset.destination === key) marker.classList.toggle('preview', active);
  }
  for (const group of document.querySelectorAll('.destination-group')) {
    if (group.dataset.destination === key) group.classList.toggle('preview', active);
  }
}

function clearSelectedDestination(focusBoard = false) {
  if (mapController.selectedDestination == null) return;
  mapController.setSelectedDestination(null);
  renderMap();
  renderActionDock();
  refreshActionAvailability();
  if (focusBoard) $('board').focus({preventScroll: true});
}

function selectNode(destination) {
  if (mapController.animationRunning || actionInFlight
      || !session.state || session.state.phase !== 'evader' || session.state.evader_mode !== 'human') return;
  const moves = session.state.legal_evader_moves.filter(move => move.destination === destination);
  if (!moves.length) return;
  if (moves.length === 1) {
    submitMove(moves[0]);
    return;
  }
  mapController.setSelectedDestination(destination);
  renderMap();
  renderActionDock();
  refreshActionAvailability();
  announce(`节点 ${destination} 可使用 ${moves.map(move => ticketLabel(move.ticket)).join('或')}，请在节点旁的选项框中选择车票。`);
  document.querySelector('#map-move-picker .map-ticket-btn')?.focus({preventScroll: true});
}

function renderPursuerTickets() {
  if (!session.state) return;
  const table = $('pursuer-ticket-table');
  table.replaceChildren();
  const header = element('div', 'pursuer-ticket-row ticket-header');
  for (const label of ['角色', 'Taxi', 'Bus', '地铁']) header.append(element('span', '', label));
  table.append(header);
  for (const [index, tickets] of (session.state.pursuer_tickets || []).entries()) {
    const row = element('div', 'pursuer-ticket-row');
    row.append(
      element('span', '', `P${index + 1}`),
      element('span', '', ticketCount(tickets.taxi, 'taxi', 'pursuer')),
      element('span', '', ticketCount(tickets.bus, 'bus', 'pursuer')),
      element('span', '', ticketCount(tickets.underground, 'underground', 'pursuer')),
    );
    table.appendChild(row);
  }
}

function renderTurn() {
  if (!session.state) return;
  if (session.state.terminal) {
    const winner = winnerPresentation(session.state, session.gameMetadata);
    $('turn-label').textContent = winner
      ? `最终回合 ${session.state.round}/${session.state.max_rounds} · ${winner} 获胜`
      : `最终回合 ${session.state.round}/${session.state.max_rounds}`;
    return;
  }
  const phaseLabel = session.state.phase === 'evader'
    ? 'Evader 行动'
    : busyKind === 'pursuer-turn' ? 'Pursuer 计算' : 'Evader 移动';
  $('turn-label').textContent = `第 ${session.state.round}/${session.state.max_rounds} 轮 · ${phaseLabel}`;
}

function renderState(previous = null) {
  if (!session.state) return Promise.resolve();
  const finalCompletedSteps = buildTurnTrack(session.state).completedSteps;
  let visibleCompletedSteps = previous
    ? Math.min(buildTurnTrack(previous).completedSteps, finalCompletedSteps)
    : finalCompletedSteps;
  setMatchLayout(true);
  cancelAnimation();
  mapController.setSelectedDestination(null);
  renderTurn();
  renderMatchCard();
  renderGameStatus(visibleCompletedSteps);
  renderActionDock();
  renderPursuerTickets();
  renderMap();
  if (session.state.terminal) stopAutoplay();
  const advanceTurnTrack = async actor => {
    if (visibleCompletedSteps >= finalCompletedSteps) return;
    const nextActor = visibleCompletedSteps % 2 === 0 ? 'evader' : 'pursuer';
    if (actor !== nextActor) return;
    visibleCompletedSteps += 1;
    renderGameStatus(visibleCompletedSteps);
    await waitForTurnTrackAnimation();
  };
  const transition = previous
    ? animateTransition(previous, session.state, advanceTurnTrack)
    : Promise.resolve();
  refreshActionAvailability();
  return transition.then(async () => {
    while (visibleCompletedSteps < finalCompletedSteps) {
      await advanceTurnTrack(visibleCompletedSteps % 2 === 0 ? 'evader' : 'pursuer');
    }
  });
}

function refreshActionAvailability() {
  const unavailable = actionInFlight || mapController.animationRunning || Boolean(session.state && !session.gameId);
  for (const button of document.querySelectorAll('.move-btn')) button.disabled = unavailable;
  $('new-seed-game').disabled = unavailable || !session.gameMetadata?.selection;
  $('new-seed-game').setAttribute('aria-busy', String(busyKind === 'start'));
  $('new-seed-game-label').textContent = busyKind === 'start' ? '创建中…' : '新对局';
  $('reset-game').disabled = unavailable || !session.gameId;
  const canUndo = undoAvailable(session.state);
  $('undo').disabled = unavailable || !canUndo;
  $('undo').title = canUndo ? '撤回上一轮' : '当前没有可以撤回的回合';
  const policyMode = Boolean(session.state && session.state.evader_mode !== 'human' && !session.state.terminal);
  $('advance').hidden = !policyMode;
  $('autoplay').hidden = !policyMode;
  $('advance').disabled = unavailable || autoplayRunning;
  $('advance').textContent = busyKind === 'advance' && busyVisual
    ? 'AZ 搜索中…'
    : '执行本轮';
  $('autoplay').disabled = unavailable && !autoplayRunning;
  $('autoplay').textContent = autoplayRunning ? '暂停播放' : '自动播放';
  $('autoplay').setAttribute('aria-pressed', String(autoplayRunning));
  syncQuickStrategies();
}

function setBusy(kind) {
  if (!kind && busyVisual) finishBusyVisual($('error').hidden);
  busyKind = kind;
  actionInFlight = Boolean(kind);
  $('app').setAttribute('aria-busy', String(actionInFlight));
  if (kind) {
    $('search-feedback').hidden = true;
    startBusyVisual(kind);
  }
  refreshStartAvailability();
  refreshActionAvailability();
  $('retry-pursuers').hidden = Boolean(kind) || session.state?.phase !== 'pursuer';
  if (!kind) {
    $('model-loading').hidden = Boolean(session.state);
    if (!session.state) {
      $('loading-detail').textContent = $('error-message').textContent || '加载尚未完成';
      $('loading-retry').hidden = false;
    }
  }
}

function applyGameResponse(result) {
  session.gameId = result.game_id || session.gameId;
  session.gameMetadata = result.metadata || session.gameMetadata;
  session.state = result;
  session.rememberGameId(session.gameId);
}

async function submitMove(move) {
  if (mapController.animationRunning || actionInFlight || !session.gameId) return false;
  clearError();
  setBusy('evader-move');
  announce(`正在使用${ticketLabel(move.ticket)}前往节点 ${move.destination}。`);
  try {
    const previous = session.state;
    const result = await api(`/api/games/${session.gameId}/evader-move`, {
      method: 'POST',
      body: JSON.stringify(move),
    });
    applyGameResponse(result);
    await renderState(previous);
    if (session.state.phase === 'pursuer' && !session.state.terminal) await requestPursuerTurn();
    announce(session.state.terminal ? `${winnerLabel(session.state.winner)}获胜。` : `已进入第 ${session.state.round} 轮。`);
    return true;
  } catch (error) {
    showError(error);
    return false;
  } finally {
    setBusy(null);
  }
}

async function requestPursuerTurn() {
  if (!session.gameId || session.state?.phase !== 'pursuer' || session.state.terminal) return;
  setBusy('pursuer-turn');
  renderTurn();
  renderActionDock();
  announce('Evader 移动完成，正在计算 Pursuer 行动。');
  const previous = session.state;
  const result = await api(`/api/games/${session.gameId}/pursuer-turn`, {
    method: 'POST',
    body: '{}',
  });
  applyGameResponse(result);
  finishBusyVisual(true);
  await renderState(previous);
}

async function advance() {
  if (mapController.animationRunning || actionInFlight || !session.gameId) return false;
  clearError();
  setBusy('advance');
  announce(`正在执行第 ${session.state.round} 轮。`);
  try {
    const previous = session.state;
    const result = await api(`/api/games/${session.gameId}/advance`, {method: 'POST', body: '{}'});
    applyGameResponse(result);
    await renderState(previous);
    return true;
  } catch (error) {
    showError(error);
    return false;
  } finally {
    setBusy(null);
  }
}

async function undo() {
  if (mapController.animationRunning || actionInFlight || !session.gameId) return;
  clearError();
  stopAutoplay();
  setBusy('undo');
  try {
    const result = await api(`/api/games/${session.gameId}/undo`, {method: 'POST', body: '{}'});
    applyGameResponse(result);
    renderState();
    announce(`已撤回到第 ${session.state.round} 轮行动前。`);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(null);
  }
}

function stopAutoplay() {
  autoplayRunning = false;
  if (autoplayTimer != null) window.clearTimeout(autoplayTimer);
  autoplayTimer = null;
  refreshActionAvailability();
}

async function autoplayStep() {
  if (!autoplayRunning || session.state?.terminal) {
    stopAutoplay();
    return;
  }
  if (mapController.animationRunning || actionInFlight) {
    autoplayTimer = window.setTimeout(autoplayStep, 80);
    return;
  }
  const ok = await advance();
  if (!ok || session.state?.terminal || !autoplayRunning) {
    stopAutoplay();
    return;
  }
  const delay = matchMedia('(prefers-reduced-motion: reduce)').matches
    ? 180
    : 140;
  autoplayTimer = window.setTimeout(autoplayStep, delay);
}

function toggleAutoplay() {
  if (autoplayRunning) {
    stopAutoplay();
    announce('自动播放已暂停。');
    return;
  }
  if (!session.state || session.state.evader_mode === 'human' || session.state.terminal) return;
  autoplayRunning = true;
  refreshActionAvailability();
  announce('自动播放已开始。');
  autoplayStep();
}

async function replaceGame(request, {busyKind: nextBusyKind, loadingMessage, successMessage}) {
  clearError();
  stopAutoplay();
  const oldGameId = session.gameId;
  setBusy(nextBusyKind);
  announce(loadingMessage);
  try {
    const result = await api('/api/games', {
      method: 'POST',
      body: JSON.stringify(request),
    });
    applyGameResponse(result);
    mapController.resetView();
    setMatchLayout(true);
    renderState();
    fitBoardToArea();
    announce(successMessage(session.gameMetadata.seed));
    if (oldGameId && oldGameId !== session.gameId) {
      api(`/api/games/${oldGameId}`, {method: 'DELETE'}).catch(() => {});
    }
    return true;
  } catch (error) {
    showError(error);
    return false;
  } finally {
    setBusy(null);
  }
}

async function start(seed = nextSeed, {
  loadingMessage = '正在加载对手。',
  successMessage = () => '对局已开始。',
} = {}) {
  if (actionInFlight || !session.catalogReady) return;
  return replaceGame(gameCreationRequest({
    ruleset: 'finite_tickets_v0',
    pursuer_strategy: $('quick-pursuer').value,
    evader_strategy: 'human',
    pursuer_action_mode: 'deterministic_argmax',
    az_simulations: 0,
  }, seed), {
    busyKind: 'start',
    loadingMessage,
    successMessage,
  });
}

async function switchPursuer() {
  if (actionInFlight || !session.catalogReady) return;
  const currentModel = session.gameMetadata?.selection.pursuer_strategy;
  const selectedModel = $('quick-pursuer').value;
  if (selectedModel === currentModel) return;
  const selectedLabel = $('quick-pursuer').selectedOptions[0]?.textContent || selectedModel;
  const switched = await start(session.gameMetadata?.seed ?? nextSeed, {
    loadingMessage: `正在切换到 ${selectedLabel}。`,
    successMessage: () => `已切换到 ${selectedLabel}，对局已重置。`,
  });
  if (!switched && currentModel) $('quick-pursuer').value = currentModel;
}

async function startNewSeedGame() {
  if (actionInFlight || !session.catalogReady) return;
  setNewSeed();
  await start();
}

async function restartGame() {
  if (!session.gameId || !session.gameMetadata || actionInFlight) return;
  clearError();
  stopAutoplay();
  setBusy('replay');
  try {
    const result = await api(`/api/games/${session.gameId}/restart`, {
      method: 'POST',
      body: JSON.stringify({seed: session.gameMetadata.seed}),
    });
    applyGameResponse(result);
    mapController.resetView();
    renderState();
    announce(`已使用 seed ${session.gameMetadata.seed} 重置对局。`);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(null);
  }
}

async function restoreGame(candidate = session.rememberedGameId(), {quiet = false} = {}) {
  if (!candidate || actionInFlight) return false;
  clearError();
  setBusy('restore');
  try {
    const result = await api(`/api/games/${candidate}/state`);
    applyGameResponse(result);
    $('quick-pursuer').value = session.gameMetadata.selection.pursuer_strategy;
    setMatchLayout(true);
    await renderState();
    fitBoardToArea();
    if (session.state.phase === 'pursuer' && session.state.evader_mode === 'human' && !session.state.terminal) {
      await requestPursuerTurn();
    }
    if (!quiet) announce(`已恢复 seed ${session.gameMetadata.seed} 的对局。`);
    return true;
  } catch (error) {
    if (error?.code === 'game_not_found') {
      const hadVisibleState = Boolean(session.state);
      session.gameId = null;
      session.rememberGameId(null);
      refreshActionAvailability();
      if (hadVisibleState || !quiet) showError(error);
    } else {
      showError(error);
    }
    return false;
  } finally {
    setBusy(null);
  }
}

async function loadCatalog() {
  session.strategyData = await api('/api/catalog');
  $('quick-pursuer').replaceChildren(...session.strategyData.pursuer.map(model => {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.id.startsWith('az-') ? 'AZ' : 'PPO GNN';
    return option;
  }));
  $('quick-pursuer').value = 'az-selfplay-000360';
  session.catalogReady = true;
}

function syncQuickStrategies() {
  $('quick-pursuer').disabled = actionInFlight || mapController.animationRunning || !session.catalogReady;
  $('new-seed-game').disabled ||= !session.catalogReady;
}

function bindEvents() {
  $('quick-pursuer').addEventListener('change', switchPursuer);
  $('error-dismiss').addEventListener('click', clearError);
  $('error-recover').addEventListener('click', () => location.reload());
  $('loading-retry').addEventListener('click', () => location.reload());
  $('retry-pursuers').addEventListener('click', async () => {
    if (actionInFlight) return;
    clearError();
    try { await requestPursuerTurn(); }
    catch (error) { showError(error); }
    finally { setBusy(null); }
  });
  $('advance').addEventListener('click', advance);
  $('autoplay').addEventListener('click', toggleAutoplay);
  $('new-seed-game').addEventListener('click', startNewSeedGame);
  $('reset-game').addEventListener('click', restartGame);
  $('undo').addEventListener('click', undo);

  const board = $('board');
  document.addEventListener('pointerdown', () => {
    document.documentElement.classList.remove('keyboard-navigation');
  }, {capture: true});
  document.addEventListener('keydown', event => {
    if (event.key === 'Tab' || event.target === board) {
      document.documentElement.classList.add('keyboard-navigation');
    }
  }, {capture: true});
  board.addEventListener('click', onMapClick);

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (document.documentElement.dataset.themeMode === 'auto') syncSiteTheme();
  });
  window.addEventListener('storage', event => {
    if (event.key === 'skz_theme') syncSiteTheme();
  });
  window.addEventListener('pageshow', event => {
    if (event.persisted && session.gameId) restoreGame(session.gameId, {quiet: true});
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && mapController.selectedDestination != null) {
      clearSelectedDestination(true);
    }
  });
}

async function boot() {
  bindEvents();
  setModelProgressListener(updateModelProgress);
  setNewSeed();
  syncSiteTheme();
  const boardResizeObserver = new ResizeObserver(fitBoardToArea);
  boardResizeObserver.observe($('layout'));
  boardResizeObserver.observe($('board-area'));
  const roundTrackResizeObserver = new ResizeObserver(layoutRoundTrack);
  roundTrackResizeObserver.observe($('round-track'));
  try {
    const [loadedMap] = await Promise.all([api('/api/map'), loadCatalog()]);
    mapController.initialize(loadedMap);
    fitBoardToArea();
    renderMap();
    renderActionDock();
    renderPursuerTickets();
    $('turn-label').textContent = '选择对手，开始逃脱';
    const restored = await restoreGame(session.rememberedGameId(), {quiet: true});
    if (!restored) await start();
  } catch (error) {
    showError(error);
    $('turn-label').textContent = '页面初始化失败';
    $('loading-detail').textContent = error.message;
    $('loading-retry').hidden = false;
  } finally {
    refreshStartAvailability();
  }
}

boot();
