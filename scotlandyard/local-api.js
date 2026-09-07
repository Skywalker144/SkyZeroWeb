import {ScotlandGame} from './engine.js';
const SAVE_KEY = 'skz_scotland_game_v1';
let board, catalog, active, worker, workerModel;
let nextId = 0;
const pending = new Map();
let progressListener = () => {};
export function setModelProgressListener(listener) { progressListener = listener; }
const asset = path => new URL(path, import.meta.url).href;
async function json(path) {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`资源加载失败（HTTP ${response.status}）`);
  return response.json();
}
const ready = Promise.all([json('./board.json'),json('./models.json')]).then(([b,c]) => {board=b;catalog=c;});
function resetWorker(error) {
  worker?.terminate(); worker=null; workerModel=null;
  for (const job of pending.values()) {clearTimeout(job.timer);job.reject(error);}
  pending.clear();
}
function rpc(message) {
  if (!worker) {
    worker = new Worker(asset('./inference-worker.js'));
    worker.onmessage = ({data}) => {
      const job = pending.get(data.id); if (!job) return;
      if (data.progress) { progressListener({...data.progress, model:job.model}); return; }
      pending.delete(data.id);clearTimeout(job.timer);
      if (data.error) job.reject(new Error(data.error)); else job.resolve(data.result);
    };
    worker.onerror = () => resetWorker(new Error('推理组件加载失败，请检查网络后重试'));
  }
  return new Promise((resolve,reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => resetWorker(new Error('模型加载或推理超时，请检查网络后重试')),120000);
    pending.set(id,{resolve,reject,timer,model:message.model});worker.postMessage({id,...message});
  });
}
async function loadModel(model) {
  if (workerModel === model.id) return;
  progressListener({model,stage:'download',loaded:0,total:model.bytes});
  await rpc({kind:'load',model,url:asset(model.url)});
  workerModel = model.id;
}
function save() {
  try { sessionStorage.setItem(SAVE_KEY,JSON.stringify({...active,game:active.game.serialize()})); } catch (_) {}
}
function snapshot() {
  return {...active.game.snapshot(), game_id:active.id, metadata:active.metadata,
    pursuer_strategy_id:active.model.id, pursuer_policy_id:active.model.label};
}
function notFound() {
  const error = new Error('保存的对局已失效，请开始新对局');error.code='game_not_found';return error;
}
async function restore(id) {
  let data;
  try { data = JSON.parse(sessionStorage.getItem(SAVE_KEY)); } catch (_) {}
  if (!data || data.id !== id) throw notFound();
  const model = catalog.pursuer.find(m => m.id === data.model.id && m.sha256 === data.model.sha256);
  if (!model) throw notFound();
  const game = ScotlandGame.restore(board,model.game,data.game);
  await loadModel(model);
  active = {...data,model,game};
}
export async function requestJson(path, options={}) {
  await ready;
  const body = JSON.parse(options.body || '{}');
  if (path === '/api/map') return board;
  if (path === '/api/catalog') return {...catalog,summary:{learned_pursuer_count:catalog.pursuer.length,learned_evader_count:0}};
  if (path === '/api/games') {
    if (body.evader_strategy !== 'human') throw new Error('本页面仅支持玩家扮演 Evader');
    if (body.ruleset !== 'finite_tickets_v0') throw new Error('模型只支持有限车票规则');
    const model = catalog.pursuer.find(m => m.id === body.pursuer_strategy);
    if (!model) throw new Error('请选择对手权重');
    const game = new ScotlandGame(board,model.game,body.seed);
    await loadModel(model);
    const id = crypto.randomUUID();
    const selection = {...body,pursuer_action_mode:'deterministic_argmax',az_simulations:0};
    active = {id,model,game,metadata:{seed:body.seed,selection,game_config:model.game,
      device:'ONNX Runtime Web · WASM',seed_protocol:'mulberry32-fold53-v1',
      opening:{evader:game.state.evader_position,pursuers:[...game.state.pursuer_positions]},
      pursuer:{kind:'onnx',name:model.label,sha256:model.sha256},evader:{kind:'human'}}};
    save();return snapshot();
  }
  const [,id,action] = path.match(/^\/api\/games\/([^/]+)(?:\/([^/]+))?$/) || [];
  if (!id) throw new Error('不支持的本地操作');
  if (options.method === 'DELETE') {if(active?.id===id){active=null;sessionStorage.removeItem(SAVE_KEY);}return {};}
  if (active?.id !== id) await restore(id);
  switch(action) {
    case 'state': return snapshot();
    case 'evader-move': active.game.evaderMove(body);break;
    case 'pursuer-turn': {
      await loadModel(active.model);
      const output = await rpc({kind:'run',inputs:active.game.publicInputs()});
      active.game.pursuerMove(output.actions,output.order);break;
    }
    case 'undo': active.game.undo();break;
    case 'restart': active.game = new ScotlandGame(board,active.model.game,active.metadata.seed);break;
    default: throw new Error('本页面不支持此操作');
  }
  save();return snapshot();
}
