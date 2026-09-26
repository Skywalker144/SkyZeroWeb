import {requestJson as api, setModelProgressListener, evaluationProgress, evaluationExport, persistenceError} from './local-api.js?v=v016-preload-1';
import {legalActionRows, ticketCountPresentation, undoAvailable} from './action_view.js';
import {gameSession as session} from './game_session.js?v=v016-blind-1';
import {createMapController} from './map_controller.js?v=v016-reveal-1';
import {matchPresentation, winnerPresentation} from './match_view.js?v=v016-pilot-1';
import {buildTurnTrack, revealPresentation} from './timeline_view.js?v=v016-reveal-1';

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
let busyKind = null;
let busyVisual = null;
const ANIMATION_STEP_MS = 100;

function element(tag, className = '', text = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== '') node.textContent = text;
  return node;
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
  const node = $('error');
  const message = friendlyError(error);
  $('error-message').textContent = message;
  $('error-recover').hidden = error?.code !== 'game_not_found';
  node.hidden = false;
  $('status-live').textContent = message;
  if (error?.code === 'game_not_found') {
    session.gameId = null;
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

function formatSeconds(ms) { return `${(ms / 1000).toFixed(1)} 秒`; }
function startBusyVisual(kind) {
  if (kind !== 'pursuer-turn') return;
  if (busyVisual) finishBusyVisual(false);
  const visual = {startedAt:performance.now(), kind};
  busyVisual = visual;
  $('search-title').textContent = 'Pursuer 正在行动';
  $('search-detail').textContent = '正在选择行动，请稍候';
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
    $('search-feedback').textContent = `对手已行动 · ${formatSeconds(elapsed)}`;
    $('search-feedback').hidden = false;
  }
}

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

function updateModelProgress({stage, loaded = 0, total = 1}) {
  $('model-loading').hidden = false;
  $('loading-retry').hidden = true;
  const percent = Math.min(100, Math.floor(loaded / total * 100));
  $('model-progress').value = percent;
  $('loading-title').textContent = stage === 'initialize' ? '正在准备对局' : '正在下载全部测评资源';
  $('loading-detail').textContent = stage === 'initialize'
    ? '全部资源已就绪，正在准备对局…'
    : `${percent}% · ${(loaded / 1048576).toFixed(2)} / ${(total / 1048576).toFixed(2)} MB${stage === 'verify' ? ' · 正在校验…' : ''}`;
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

function terminalReasonLabel(reason) {
  return {
    capture: 'Pursuer 完成抓捕',
    max_rounds: 'Evader 存活至回合上限',
    evader_stuck: 'Evader 无合法动作',
    pursuers_stuck: '所有 Pursuer 均无合法动作',
    give_up: '已主动认输',
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
  track.replaceChildren(...Array.from({length:progress.totalSteps/2},(_,index) => {
    const [evader,pursuer] = progress.steps.slice(index*2,index*2+2);
    const cell = element('span','round-cell');
    cell.dataset.round = String(index+1);
    cell.classList.toggle('complete',pursuer.played);
    cell.classList.toggle('active',!progress.terminal && (evader.current || pursuer.current));
    cell.classList.toggle('scheduled',evader.scheduled);
    cell.classList.toggle('exposed',evader.exposedAt !== undefined);
    cell.append(element('b','',index+1),element('small','',evader.exposedAt !== undefined ? '已揭示' : evader.scheduled ? '待揭示' : ''));
    cell.title = `第 ${index+1} 回合` + (evader.exposedAt !== undefined
      ? ` · 移动后已公开节点 ${evader.exposedAt}` : evader.scheduled ? ' · 逃跑者移动后公开落点' : '');
    cell.setAttribute('role','listitem');cell.setAttribute('aria-label',cell.title);
    if(cell.classList.contains('active')) cell.setAttribute('aria-current','step');
    return cell;
  }));
  track.setAttribute('role','list');
  track.setAttribute('aria-label','回合进度；揭示发生在逃跑者移动后');
}

function renderReveal() {
  const state = session.displayState ?? session.state;
  const {revealsNow,nextRound} = revealPresentation(state);
  $('reveal-hint').textContent = revealsNow
    ? `本次移动后公开落点！第 ${state.round} 回合使用黑票也会揭示位置。`
    : state?.terminal ? '当前尝试已结束。'
      : nextRound ? `下一次揭示：第 ${nextRound} 回合移动后。` : '本局没有后续揭示回合。';
  $('reveal-hint').classList.toggle('reveal-now',revealsNow);
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
    list.append(element('span', 'empty', '点击开始测评，开始逃脱。'));
    return;
  }
  if (session.state.terminal) {
    const retry = session.state.challenge.result === 'in_progress';
    $('actions-title').textContent = retry ? '可以悔棋继续' : '挑战已结算';
    $('actions-hint').textContent = retry ? '本次失利尚未计入失败' : '本局已记录，可继续下一局';
    list.append(element('span', 'empty', retry ? '点击“悔棋继续”尝试其他走法，或“认输结束”结算本局。' : '本局成绩已记录。'));
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
  const state = session.displayState ?? session.state;
  if (!state) return;
  if (state.terminal) {
    if (state.challenge.result === 'in_progress') {
      $('turn-label').textContent = `第 ${state.round}/${state.max_rounds} 轮 · ${terminalReasonLabel(state.terminal_reason)}，可悔棋继续`;
      return;
    }
    if (state.challenge.result === 'give_up') {
      $('turn-label').textContent = `已认输 · 本局计为失败 · 累计悔棋 ${state.challenge.undo_num} 次`;
      return;
    }
    const winner = winnerPresentation(state, session.gameMetadata);
    $('turn-label').textContent = winner
      ? `最终回合 ${state.round}/${state.max_rounds} · ${winner} 获胜`
      : `最终回合 ${state.round}/${state.max_rounds}`;
    return;
  }
  const phaseLabel = state.phase === 'evader'
    ? 'Evader 行动'
    : busyKind === 'pursuer-turn' ? 'Pursuer 行动' : 'Evader 移动';
  $('turn-label').textContent = `第 ${state.round}/${state.max_rounds} 轮 · ${phaseLabel}`;
}

function renderEvaluation() {
  const progress = evaluationProgress();
  $('evaluation-start').textContent = progress.batch_id ? '继续测评' : '开始测评';
  $('evaluation-heading').textContent = progress.complete
    ? `第 ${progress.batch_number} 组测评已完成`
    : `第 ${progress.batch_number} 组 · 第 ${progress.game_number}/${progress.total} 局`;
  $('evaluation-progress').max = progress.total;
  $('evaluation-progress').value = progress.completed;
  $('evaluation-progress-text').textContent = `已完成 ${progress.completed}/${progress.total} 局 · 本组成功 ${progress.wins} 局`;
  $('evaluation-total').textContent = `累计完成 ${progress.num_completed_batches} 组测评`;
  $('evaluation-repeat').hidden = !progress.complete;
  $('evaluation-next').hidden = !progress.can_next;
  const message = persistenceError() || '进度已保存在本机，关闭页面后可继续。';
  for(const node of document.querySelectorAll('[data-storage-status]')) {
    node.textContent = message;node.classList.toggle('storage-error',Boolean(persistenceError()));
  }
  const challenge = session.state?.challenge;
  $('undo-count').textContent = `本局累计悔棋 ${challenge?.undo_num ?? 0} 次`;
  $('evaluation-current').textContent = progress.complete
    ? '20 局已完成！请导出 JSON。你也可以再测一组，新结果会另行保存。'
    : challenge?.result === 'in_progress' ? '允许无限悔棋；成功或认输才结算本局。'
      : challenge?.result === 'win' ? '逃脱成功！本局已记录。' : challenge ? '已认输，本局已记录。' : '';
}

async function renderState(previous = null) {
  if (!session.state) return;
  session.displayState = previous ?? session.state;
  renderEvaluation();
  setMatchLayout(true);
  cancelAnimation();
  mapController.setSelectedDestination(null);
  renderTurn();renderReveal();renderMatchCard();
  renderGameStatus(previous ? buildTurnTrack(previous).completedSteps : null);
  renderActionDock();renderPursuerTickets();renderMap();
  refreshActionAvailability();
  if (previous) await animateTransition(previous,session.state);
  session.displayState = session.state;
  renderTurn();renderReveal();renderGameStatus();renderMap();
  refreshActionAvailability();
}

function refreshActionAvailability() {
  const unavailable = actionInFlight || mapController.animationRunning;
  for (const button of document.querySelectorAll('.move-btn')) button.disabled = unavailable;
  const progress = evaluationProgress();
  $('evaluation-start').disabled = unavailable || !session.catalogReady;
  $('evaluation-repeat').disabled = unavailable || !progress.complete;
  $('evaluation-next').disabled = unavailable || !progress.can_next;
  const canUndo = undoAvailable(session.state);
  $('reset-game').disabled = unavailable || !canUndo;
  $('give-up').disabled = unavailable || session.state?.challenge?.result !== 'in_progress';
  $('undo').disabled = unavailable || !canUndo;
  $('undo').textContent = session.state?.terminal && canUndo ? '悔棋继续' : '悔棋一轮';
  $('undo').title = canUndo ? '退回一轮，累计悔棋次数加 1' : '当前没有可以悔棋的回合';
}

function setBusy(kind) {
  if (!kind && busyVisual) finishBusyVisual($('error').hidden);
  busyKind = kind;
  actionInFlight = Boolean(kind);
  $('app').setAttribute('aria-busy',String(actionInFlight));
  if (kind) { $('search-feedback').hidden = true;startBusyVisual(kind); }
  refreshActionAvailability();
  $('retry-pursuers').hidden = Boolean(kind) || session.state?.phase !== 'pursuer' || session.state?.terminal;
  if (!kind) $('model-loading').hidden = true;
}

function applyGameResponse(result) {
  session.gameId = result.game_id || session.gameId;
  session.gameMetadata = result.metadata || session.gameMetadata;
  session.state = result;
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
    announce(session.state.terminal ? $('actions-hint').textContent : `已进入第 ${session.state.round} 轮。`);
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

async function undo() {
  if (mapController.animationRunning || actionInFlight || !session.gameId) return;
  clearError();
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

async function loadEvaluation(action) {
  if (actionInFlight || !session.catalogReady) return;
  clearError();setBusy('start');
  try {
    applyGameResponse(await api(`/api/evaluation/${action}`,{method:'POST'}));
    mapController.resetView();
    await renderState();fitBoardToArea();
    if (session.state.phase === 'pursuer' && !session.state.terminal) await requestPursuerTurn();
    announce('测评已就绪，请选择目的地。');
  } catch (error) { showError(error); }
  finally { renderEvaluation();setBusy(null); }
}

async function restartGame() {
  if (!session.gameId || !session.gameMetadata || actionInFlight) return;
  clearError();
  setBusy('replay');
  try {
    const result = await api(`/api/games/${session.gameId}/restart`, {
      method: 'POST',
      body: '{}',
    });
    applyGameResponse(result);
    mapController.resetView();
    renderState();
    announce(`已退回开局，本局累计悔棋 ${session.state.challenge.undo_num} 次。`);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(null);
  }
}

async function giveUp() {
  if (actionInFlight || mapController.animationRunning || session.state?.challenge?.result !== 'in_progress') return;
  clearError();setBusy('give-up');
  try {
    applyGameResponse(await api(`/api/games/${session.gameId}/give-up`, {method:'POST',body:'{}'}));
    await renderState();
    announce('已认输，本局计为失败，累计悔棋次数已保存。');
  } catch (error) { showError(error); }
  finally { setBusy(null); }
}

async function restoreGame(candidate, {quiet = false} = {}) {
  if (!candidate || actionInFlight) return false;
  clearError();
  setBusy('restore');
  try {
    const result = await api(`/api/games/${candidate}/state`);
    applyGameResponse(result);
    setMatchLayout(true);
    await renderState();
    fitBoardToArea();
    if (session.state.phase === 'pursuer' && session.state.evader_mode === 'human' && !session.state.terminal) {
      await requestPursuerTurn();
    }
    if (!quiet) announce('已恢复当前测评。');
    return true;
  } catch (error) {
    if (error?.code === 'game_not_found') {
      const hadVisibleState = Boolean(session.state);
      session.gameId = null;
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
  const rules = session.strategyData.default_game;
  for(const node of document.querySelectorAll('[data-rules-copy]')) node.textContent =
    `你扮演逃跑者，对抗 ${rules.num_pursuers} 名追捕者。第 ${rules.reveal_rounds.join('、')} 回合移动后公开落点，黑票也不例外；最多 ${rules.max_rounds} 回合。普通车票不限，${rules.resolved_ruleset.evader_initial_tickets[3]} 张黑票可隐藏交通方式或坐船。`;
  session.catalogReady = true;
}

function bindEvents() {
  for(const button of document.querySelectorAll('[data-export]')) button.addEventListener('click', () => {
    const data = evaluationExport();
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = `scotland-evaluation-${data.participant_id.slice(0,8)}-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  $('evaluation-start').addEventListener('click',() => loadEvaluation('start'));
  $('evaluation-repeat').addEventListener('click',() => loadEvaluation('start'));
  $('evaluation-next').addEventListener('click',() => loadEvaluation('next'));
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
  $('reset-game').addEventListener('click', restartGame);
  $('undo').addEventListener('click', undo);
  $('give-up').addEventListener('click', giveUp);

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
  syncSiteTheme();
  setBusy(null);
  const boardResizeObserver = new ResizeObserver(fitBoardToArea);
  boardResizeObserver.observe($('layout'));
  boardResizeObserver.observe($('board-area'));
  try {
    const [loadedMap] = await Promise.all([api('/api/map'), loadCatalog()]);
    mapController.initialize(loadedMap);
    fitBoardToArea();
    renderMap();
    renderActionDock();
    renderPursuerTickets();
    $('turn-label').textContent = '开始测评后显示回合';
    if (session.strategyData.active_game_id) await restoreGame(session.strategyData.active_game_id,{quiet:true});
    renderEvaluation();
  } catch (error) {
    showError(error);
    $('turn-label').textContent = '页面初始化失败';
    $('error-recover').hidden = false;
  } finally {
    refreshActionAvailability();
  }
}

boot();
