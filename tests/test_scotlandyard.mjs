import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ScotlandGame} from '../public/scotlandyard/engine.js';
import {matchPresentation} from '../public/scotlandyard/match_view.js';
const board=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/board.json',import.meta.url)));
const {default_game:config}=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/models.json',import.meta.url)));
const game=seed=>new ScotlandGame(board,config,seed);
function pending(g){g.evaderMove(g.legalEvader()[0]);return g;}
test('compact controller labels identify AZ and PPO pursuers',()=>{
 assert.equal(matchPresentation({}, {pursuer:{kind:'onnx',name:'AZ Selfplay · 330（默认）'},evader:{kind:'human'}}).pursuer,'AZ Pursuer');
 assert.equal(matchPresentation({}, {pursuer:{kind:'onnx',name:'PPO GNN · 500'},evader:{kind:'human'}}).pursuer,'PPO Pursuer');
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
test('finite tickets require forced pass, preserve inventory, support undo/restore',()=>{
 const g=game(42);
 g.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 const before=g.snapshot();pending(g);
 const midway=g.snapshot();assert.throws(()=>g.pursuerMove([0,-1,-1],[0,1,2]));assert.deepEqual(g.snapshot(),midway);
 g.pursuerMove([-1,-1,-1],[2,0,1]);
 assert.deepEqual(g.state.pursuer_positions,before.pursuer_positions);
 const restored=ScotlandGame.restore(board,config,JSON.parse(JSON.stringify(g.serialize())));
 restored.undo();assert.deepEqual(restored.snapshot(),before);
});
test('23rd-round capture takes priority over Evader survival',()=>{
 const g=pending(game(42));g.state.round=23;
 const rows=g.candidateRows(),positions=[...g.state.pursuer_positions],actions=[];
 for(let i=0;i<3;i++){
   const available=rows[i].findIndex(m=>!positions.some((n,j)=>j!==i&&n===m.destination));
   actions[i]=available;if(available>=0)positions[i]=rows[i][available].destination;
 }
 g.state.evader_position=positions[0];g.pursuerMove(actions,[0,1,2]);
 assert.equal(g.state.terminal_reason,'capture');assert.equal(g.state.winner,'pursuers');
});
test('public reveals, all-pass round cap, and browser seed replay',()=>{
 const g=game(81);g.state.pursuer_tickets.forEach(t=>{t.taxi=t.bus=t.underground=0;});
 for(let round=1;round<=23;round++){
   pending(g);
   if(config.reveal_rounds.includes(round))assert.deepEqual(g.state.possible_locations,[g.state.evader_position]);
   g.pursuerMove([-1,-1,-1],[0,1,2]);
 }
 assert.equal(g.state.terminal_reason,'max_rounds');assert.equal(g.state.winner,'evader');
 assert.deepEqual(game(9007199254740991).snapshot(),game(9007199254740991).snapshot());
 assert.notDeepEqual(game(42).state.pursuer_positions,game(42+4294967296).state.pursuer_positions);
});
