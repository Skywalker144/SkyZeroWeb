import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {PilotChallenge} from '../public/scotlandyard/pilot-challenge.js';
const board=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/board.json',import.meta.url)));
const catalog=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/models.json',import.meta.url)));
const model=catalog.pursuer[1];
const fresh=()=>new PilotChallenge(board,model.game,42);
function move(c) { c.evaderMove(c.snapshot().legal_evader_moves[0]); }
function capture(c) {
 move(c);
 c.game.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 c.game.state.evader_position=c.state.pursuer_positions[0];
 c.pursuerMove([-1,-1,-1,-1,-1],[0,1,2,3,4]);
}
function win(c) {
 move(c);
 c.game.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 c.pursuerMove([-1,-1,-1,-1,-1],[0,1,2,3,4]);
}
test('capture is provisional; repeated retries accumulate undo outside restored states',()=>{
 const c=fresh(),initial=c.snapshot();
 for(let i=1;i<=7;i++) {
  capture(c);assert.equal(c.result,'in_progress');assert.equal(c.snapshot().undo_available,true);
  c.undo();assert.equal(c.undoCount,i);assert.equal(c.state.terminal,false);
  assert.deepEqual(c.state.pursuer_tickets,initial.pursuer_tickets);
  assert.equal(c.gameLength,0);
 }
 assert.throws(()=>c.undo());assert.equal(c.undoCount,7);
});
test('rewind counts the actual rollback depth and retains the same challenge',()=>{
 const c=fresh();move(c);c.undo();move(c);
 const positions=[...c.state.pursuer_positions];
 const actions=c.game.candidateRows().map((row,i)=>{
  const a=row.findIndex(m=>!positions.some((n,j)=>i!==j&&n===m.destination));
  if(a>=0)positions[i]=row[a].destination;return a;
 });
 c.pursuerMove(actions,[0,1,2,3,4]);
 assert.equal(c.state.terminal,false);move(c);
 assert.equal(c.gameLength,2);c.rewind();
 assert.equal(c.undoCount,3);assert.equal(c.gameLength,0);assert.equal(c.state.round,1);
 assert.throws(()=>c.rewind());assert.equal(c.undoCount,3);
});
test('snapshot and restore retain counters and pending AI turns without exporting them',()=>{
 const c=fresh();capture(c);c.undo();move(c);
 const restored=PilotChallenge.restore(board,model.game,JSON.parse(JSON.stringify(c.serialize())));
 assert.deepEqual(restored.snapshot(),c.snapshot());restored.undo();assert.equal(restored.undoCount,2);
});
test('success settles once; give-up locks human and AI actions and undo',()=>{
 for(const outcome of ['win','give_up']) {
  const c=fresh();if(outcome==='win')win(c);else {move(c);c.giveUp();}
  assert.equal(c.result,outcome);assert.equal(c.snapshot().terminal,true);
  assert.deepEqual(c.snapshot().legal_evader_moves,[]);assert.equal(c.snapshot().undo_available,false);
  assert.throws(()=>c.undo());assert.throws(()=>c.rewind());assert.throws(()=>c.evaderMove({}));
  assert.throws(()=>c.pursuerMove([],[]));assert.throws(()=>c.giveUp());
 }
});
