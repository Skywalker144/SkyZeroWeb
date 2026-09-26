import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ScotlandGame} from '../public/scotlandyard/engine.js';
const board=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/board.json',import.meta.url)));
const {default_game:config}=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/models.json',import.meta.url)));
const game=seed=>new ScotlandGame(board,config,seed);
const pending=g=>{g.evaderMove(g.legalEvader()[0]);return g;};
function action(g) {
 const positions=[...g.state.pursuer_positions];
 const actions=g.candidateRows().map((row,i)=>{
  const selected=row.findIndex(m=>!positions.some((n,j)=>j!==i&&n===m.destination));
  if(selected>=0)positions[i]=row[selected].destination;
  return selected;
 });
 return {actions,order:positions.map((_,i)=>i),positions};
}
test('published game uses V0.16 five-pursuer rules',()=>{
 assert.equal(config.num_pursuers,5);
 assert.equal(config.max_rounds,22);
 assert.equal(config.resolved_ruleset.terminal,'capture-pursuers-stuck-round-cap-evader-stuck-v2');
 assert.equal(config.resolved_ruleset.black_ticket_mode,'hidden');
 assert.equal(game(42).state.pursuer_positions.length,5);
});
test('invalid moves are atomic and private Evader location is absent from model inputs',()=>{
 const g=game(42),before=g.snapshot();
 assert.throws(()=>g.evaderMove({destination:0,ticket:'black'}));
 assert.deepEqual(g.snapshot(),before);
 pending(g);const inputs=g.publicInputs();
 const alternatives=g.state.possible_locations.filter(n=>n!==g.state.evader_position);
 assert.ok(alternatives.length);g.state.evader_position=alternatives[0];
 assert.deepEqual(g.publicInputs(),inputs);
});
test('hidden black tickets reach every ordinary destination',()=>{
 const g=game(81),moves=g.legalEvader();
 for(const move of moves.filter(m=>m.ticket!=='black'))
  assert.ok(moves.some(m=>m.ticket==='black'&&m.destination===move.destination));
 const move=moves.find(m=>m.ticket==='black');g.evaderMove(move);
 assert.ok(g.state.possible_locations.includes(move.destination));
 assert.equal(g.state.evader_tickets.black,4);
});
test('all detectives stuck ends after the joint action, before the round cap',()=>{
 const g=pending(game(42));
 g.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 const midway=g.snapshot();
 assert.throws(()=>g.pursuerMove([0,-1,-1,-1,-1],[0,1,2,3,4]));
 assert.deepEqual(g.snapshot(),midway);
 g.pursuerMove([-1,-1,-1,-1,-1],[4,3,2,1,0]);
 assert.equal(g.state.terminal_reason,'pursuers_stuck');
 assert.equal(g.state.winner,'evader');assert.equal(g.state.round,1);
 const restored=ScotlandGame.restore(board,config,JSON.parse(JSON.stringify(g.serialize())));
 assert.deepEqual(restored.snapshot(),g.snapshot());
 restored.undo();assert.equal(restored.state.terminal,false);
});
test('capture wins over pursuers stuck and the round cap',()=>{
 const g=pending(game(42));g.state.round=22;
 g.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 g.state.evader_position=g.state.pursuer_positions[0];
 g.pursuerMove([-1,-1,-1,-1,-1],[0,1,2,3,4]);
 assert.equal(g.state.terminal_reason,'capture');assert.equal(g.state.winner,'pursuers');
});
test('round cap and public reveals follow the exported rules',()=>{
 const g=pending(game(81));g.state.round=22;const a=action(g);
 g.state.evader_position=board.start_nodes.find(n=>!a.positions.includes(n));
 g.pursuerMove(a.actions,a.order);assert.equal(g.state.terminal_reason,'max_rounds');
 const r=game(42);r.state.round=3;pending(r);
 assert.deepEqual(r.state.possible_locations,[r.state.evader_position]);
});
test('seed replay and serialized state preserve the exact game',()=>{
 assert.deepEqual(game(9007199254740991).snapshot(),game(9007199254740991).snapshot());
 assert.notDeepEqual(game(42).state.pursuer_positions,game(42+4294967296).state.pursuer_positions);
 const g=pending(game(42));
 assert.deepEqual(ScotlandGame.restore(board,config,JSON.parse(JSON.stringify(g.serialize()))).snapshot(),g.snapshot());
});
