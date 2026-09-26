import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ScotlandGame} from '../public/scotlandyard/engine.js';
import {buildTurnTrack, revealPresentation} from '../public/scotlandyard/timeline_view.js';

const board=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/board.json',import.meta.url)));
const config=JSON.parse(fs.readFileSync(new URL('../public/scotlandyard/models.json',import.meta.url))).default_game;

test('reveal prompts follow each actual move boundary and black tickets do not delay disclosure',()=>{
  for(const round of config.reveal_rounds) {
    const game=new ScotlandGame(board,config,42);game.state.round=round;
    assert.equal(revealPresentation(game.snapshot()).revealsNow,true);
    const move=game.legalEvader().find(m=>m.ticket==='black');game.evaderMove(move);
    const revealed=revealPresentation(game.snapshot());
    assert.equal(revealed.revealsNow,false);assert.equal(revealed.current.destination,move.destination);
    assert.equal(revealed.current.round,round);
    assert.deepEqual(game.publicInputs().possible_mask.flatMap((v,i)=>v?[i]:[]),[move.destination]);
    game.undo();assert.equal(revealPresentation(game.snapshot()).current,null);
    assert.equal(revealPresentation(game.snapshot()).revealsNow,true);
  }
});

test('turn animation never presents a future disclosure as already revealed',()=>{
  const state={max_rounds:22,reveal_rounds:[3,8,13,18],round:3,phase:'pursuer',
    evader_history:Array(3).fill({}),round_history:Array(2).fill({}),exposed_positions:[{round:3,destination:115}]};
  assert.equal(buildTurnTrack(state,4).steps[4].exposedAt,undefined);
  assert.equal(buildTurnTrack(state,5).steps[4].exposedAt,115);
  assert.equal(buildTurnTrack(state,4).steps[4].scheduled,true);
  assert.deepEqual(revealPresentation(state).current,{round:3,destination:115});
  assert.equal(revealPresentation({...state,round:4,phase:'evader'}).current,null);
  assert.equal(revealPresentation({...state,round:4,phase:'evader'}).nextRound,8);
  assert.equal(revealPresentation({...state,round:19,phase:'evader'}).nextRound,null);
});
