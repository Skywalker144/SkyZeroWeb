import {PilotChallenge} from './pilot-challenge.js?v=v016-undo-1';
import {EvaluationStore, modelKey} from './evaluation-store.js?v=v016-blind-1';
let storage;
try { storage = globalThis.localStorage; } catch (_) {}
const evaluation = new EvaluationStore(storage);
export const evaluationProgress = () => evaluation.progress();
export const evaluationExport = () => evaluation.export();
export const persistenceError = () => evaluation.error;
let board, catalog, active, worker, workerModel;
let workerPrepared = false;
let nextId = 0;
let mutating = false;
const pending = new Map();
let progressListener = () => {};
export function setModelProgressListener(listener) { progressListener = listener; }
const asset = path => new URL(path, import.meta.url).href;
async function json(path) {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`资源加载失败（HTTP ${response.status}）`);
  return response.json();
}
const ready = Promise.all([json('./board.json?v=v016-pilot-1'),json('./models.json?v=v016-pilot-1')]).then(([b,c]) => {board=b;catalog=c;});
function resetWorker(error) {
  worker?.terminate(); worker=null; workerModel=null;
  workerPrepared = false;
  for (const job of pending.values()) {clearTimeout(job.timer);job.reject(error);}
  pending.clear();
}
function refreshTimeout(job) {
  clearTimeout(job.timer);
  job.timer = setTimeout(() => resetWorker(new Error('资源加载或推理超时，请检查网络后重试')),120000);
}
function rpc(message) {
  if (!worker) {
    worker = new Worker(asset(`./inference-worker.js?v=${Date.now()}`));
    worker.onmessage = ({data}) => {
      const job = pending.get(data.id); if (!job) return;
      if (data.progress) { refreshTimeout(job);progressListener(data.progress);return; }
      pending.delete(data.id);clearTimeout(job.timer);
      if (data.error) job.reject(new Error(data.error)); else job.resolve(data.result);
    };
    worker.onerror = () => resetWorker(new Error('推理组件加载失败，请检查网络后重试'));
  }
  return new Promise((resolve,reject) => {
    const id = ++nextId;
    const job = {resolve,reject,timer:null};
    refreshTimeout(job);
    pending.set(id,job);worker.postMessage({id,...message});
  });
}
async function loadModel(model) {
  if (workerModel === model.id) return;
  const total = catalog.pursuer.reduce((sum, item) => sum + item.bytes, 0);
  if (!workerPrepared) {
    progressListener({stage:'download',loaded:0,total});
    await rpc({kind:'prepare',models:catalog.pursuer.map(model => ({model,url:asset(model.url)}))});
    workerPrepared = true;
  }
  progressListener({stage:'initialize',loaded:total,total});
  await rpc({kind:'load',model});
  workerModel = model.id;
}
function snapshot() {
  return {...active.game.snapshot(),game_id:active.id,
    metadata:{game_config:active.model.game,pursuer:{kind:'onnx',name:'AI'},evader:{kind:'human'}},
    pursuer_strategy_id:'blind',pursuer_policy_id:'AI'};
}
async function restore(id) {
  const data = evaluation.active();
  if (!data || data.id !== id) throw new Error('当前对局已更新，请刷新页面恢复。');
  const model = catalog.pursuer.find(m => modelKey(m) === data.model_id);
  if (!model) throw new Error('保存的测评与当前资源不匹配，请联系维护者。');
  const game = PilotChallenge.restore(board,model.game,data.game);
  if (!game.settled) await loadModel(model);
  active = {id:data.id,model,game};
}
async function openNext() {
  const slot = evaluation.nextSlot();
  if (!slot) throw new Error('本组测评已经完成。');
  if (slot.result === 'in_progress') {
    await restore(slot.game_id);
  } else {
    const model = catalog.pursuer.find(m => modelKey(m) === slot.model_id);
    if (!model) throw new Error('测评资源不完整，请联系维护者。');
    const next = {id:slot.game_id,model,game:new PilotChallenge(board,model.game,slot.seed)};
    await loadModel(model);
    evaluation.save(next);
    active = next;
  }
  return snapshot();
}
export async function requestJson(path, options={}) {
  await ready;
  if (path === '/api/map') return board;
  if (path === '/api/catalog') return {default_game:catalog.default_game,active_game_id:evaluation.active()?.id};
  if (mutating) throw new Error('请等待当前操作完成。');
  mutating = true;
  try {
    evaluation.assertWritable();
    if (path === '/api/evaluation/start') {
      evaluation.ensureBatch(catalog.pursuer);
      return await openNext();
    }
    if (path === '/api/evaluation/next') {
      if (!evaluation.progress().can_next) throw new Error('请先完成当前对局。');
      return await openNext();
    }
    const [,id,action] = path.match(/^\/api\/games\/([^/]+)\/([^/]+)$/) || [];
    if (!id) throw new Error('不支持的测评操作');
    if (active?.id !== id) await restore(id);
    const body = JSON.parse(options.body || '{}');
    switch(action) {
      case 'state': return snapshot();
      case 'evader-move': active.game.evaderMove(body);break;
      case 'pursuer-turn': {
        await loadModel(active.model);
        const output = await rpc({kind:'run',inputs:active.game.publicInputs()});
        evaluation.assertWritable();
        active.game.pursuerMove(output.actions,output.order);break;
      }
      case 'undo': active.game.undo();break;
      case 'restart': active.game.rewind();break;
      case 'give-up': active.game.giveUp();break;
      default: throw new Error('不支持的测评操作');
    }
    evaluation.save(active);
    return snapshot();
  } finally { mutating = false; }
}
