import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {EvaluationStore, EVALUATION_STORAGE_KEY, EVALUATION_QUOTAS} from '../public/scotlandyard/evaluation-store.js';
import {PilotChallenge} from '../public/scotlandyard/pilot-challenge.js';

const board=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/board.json',import.meta.url)));
const catalog=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/models.json',import.meta.url)));
const disk=()=>{const values=new Map();return {getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,v)};};
function start(s) {
  const slot=s.nextSlot(),model=catalog.pursuer.find(m=>m.checkpoint===s.data.models[slot.model_id].checkpoint);
  const active={id:slot.game_id,model,game:new PilotChallenge(board,model.game,slot.seed)};
  s.save(active);return active;
}

test('each batch has exact quotas; its random order and openings survive restart',()=>{
  const storage=disk(),s=new EvaluationStore(storage);
  assert.equal(s.progress().total,20);assert.equal(s.progress().batch_id,null);
  s.ensureBatch(catalog.pursuer);
  const batch=s.data.batches[0];
  assert.equal(batch.games.length,20);
  const counts={};for(const g of batch.games) {const cp=s.data.models[g.model_id].checkpoint;counts[cp]=(counts[cp]||0)+1;}
  assert.deepEqual(counts,EVALUATION_QUOTAS);
  assert.equal(new Set(batch.games.map(g=>g.game_id)).size,20);
  s.ensureBatch(catalog.pursuer);assert.equal(s.data.batches.length,1);
  const restored=new EvaluationStore(storage);assert.deepEqual(restored.data,s.data);
  const orders=new Set();
  for(let i=0;i<8;i++) {const x=new EvaluationStore(disk());x.ensureBatch(catalog.pursuer);orders.add(x.data.batches[0].games.map(g=>g.model_id).join());}
  assert.ok(orders.size>1);
});

test('unfinished challenges cannot be skipped, replaced or counted; settled games cannot change',()=>{
  const s=new EvaluationStore(disk());s.ensureBatch(catalog.pursuer);const a=start(s);
  a.game.evaderMove(a.game.snapshot().legal_evader_moves[0]);s.save(a);
  assert.equal(s.progress().completed,0);assert.equal(s.nextSlot().game_id,a.id);
  assert.throws(()=>s.save({...a,id:s.data.batches[0].games[1].game_id}));
  a.game.undo();s.save(a);a.game.giveUp();s.save(a);s.save(a);
  assert.equal(s.progress().completed,1);assert.equal(s.progress().can_next,true);
  const settled=s.export().batches[0].games[0];assert.equal(settled.undo_num,1);assert.equal(settled.result,'give_up');
  a.game.undoCount=0;assert.throws(()=>s.save(a));assert.deepEqual(s.export().batches[0].games[0],settled);
});

test('twenty settlements finish one batch and a repeat creates fresh ids without erasing results',()=>{
  const storage=disk(),s=new EvaluationStore(storage);s.ensureBatch(catalog.pursuer);
  for(let i=0;i<20;i++) {const a=start(s);a.game.giveUp();s.save(a);assert.equal(s.progress().completed,i+1);}
  assert.equal(s.progress().complete,true);assert.equal(s.nextSlot(),null);
  const old=s.export().batches[0];const participant=s.data.participant_id;
  s.ensureBatch(catalog.pursuer);
  assert.equal(s.progress().batch_number,2);assert.equal(s.progress().completed,0);assert.equal(s.active(),null);
  assert.equal(s.data.participant_id,participant);assert.deepEqual(s.export().batches[0],old);
  assert.ok(s.data.batches[1].games.every(g=>!old.games.some(p=>p.game_id===g.game_id)));
  assert.deepEqual(new EvaluationStore(storage).data,s.data);
});

test('mid-batch exports contain started games only and summaries can be deduplicated by stable ids',()=>{
  const s=new EvaluationStore(disk());s.ensureBatch(catalog.pursuer);
  const a=start(s);a.game.evaderMove(a.game.snapshot().legal_evader_moves[0]);a.game.undo();a.game.giveUp();s.save(a);
  const b=start(s);const exported=s.export();
  assert.equal(exported.batches[0].games.length,2);assert.equal(exported.batches[0].status,'in_progress');
  assert.equal(exported.summary.reduce((sum,r)=>sum+r.num_games,0),1);
  assert.equal(exported.summary.reduce((sum,r)=>sum+r.total_undo_num,0),1);
  assert.equal(exported.summary.reduce((sum,r)=>sum+r.num_in_progress,0),1);
  assert.deepEqual(s.export().batches[0].games,exported.batches[0].games);
  assert.equal(exported.batches[0].games[1].game_id,b.id);
  const raw=JSON.stringify(exported);
  for(const field of ['undoStack','possible_locations','seed','opening','active','metadata']) assert.ok(!raw.includes(`"${field}"`));
  assert.ok(raw.length<12000);
});

test('old pilot data stays separate and concurrent stale tabs cannot overwrite progress',()=>{
  const storage=disk();storage.setItem('skz_scotland_v016_unlimited_undo_v1','old results');
  const s=new EvaluationStore(storage);s.ensureBatch(catalog.pursuer);
  const stale=new EvaluationStore(storage),a=start(s);a.game.giveUp();s.save(a);
  assert.throws(()=>stale.assertWritable(),/页面|刷新/);
  assert.equal(storage.getItem('skz_scotland_v016_unlimited_undo_v1'),'old results');
  const corrupt=disk();corrupt.setItem(EVALUATION_STORAGE_KEY,'broken');
  const blocked=new EvaluationStore(corrupt);assert.throws(()=>blocked.ensureBatch(catalog.pursuer));
  assert.equal(corrupt.getItem(EVALUATION_STORAGE_KEY),'broken');
  const unavailable=new EvaluationStore({getItem:()=>null,setItem:()=>{throw Error('quota');}});
  unavailable.ensureBatch(catalog.pursuer);start(unavailable);assert.match(unavailable.error,/导出/);
  assert.equal(unavailable.export().batches[0].games.length,1);
});
