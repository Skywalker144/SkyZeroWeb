import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ScotlandGame} from '../scotlandyard/engine.js';
const board = JSON.parse(fs.readFileSync(new URL('../scotlandyard/board.json',import.meta.url)));
const {default_game:config} = JSON.parse(fs.readFileSync(new URL('../scotlandyard/models.json',import.meta.url)));
let count=0;
for (const fixture of JSON.parse(fs.readFileSync(process.argv[2]))) {
  const game = new ScotlandGame(board,config,fixture.seed,fixture.opening);
  for (const step of fixture.steps) {
    if(step.kind==='evader') game.evaderMove(step.move);
    else game.pursuerMove(step.actions,step.order);
    const snap=game.snapshot();
    for(const [key,value] of Object.entries(step.state)) assert.deepEqual(snap[key],value,`seed ${fixture.seed}, step ${count}, ${key}`);
    if(step.inputs) {
      const actual=game.publicInputs();
      for(const [key,value] of Object.entries(step.inputs)) {
        assert.equal(actual[key].length,value.length);
        actual[key].forEach((v,i)=>assert.ok(Math.abs(Number(v)-Number(value[i])) < 1e-7,`input ${key}[${i}]`));
      }
    }
    count++;
  }
  const restored=ScotlandGame.restore(board,config,JSON.parse(JSON.stringify(game.serialize())));
  assert.deepEqual(restored.snapshot(),game.snapshot());
  game.undo();restored.undo();assert.deepEqual(restored.snapshot(),game.snapshot());
}
console.log(`${count} browser/Python state transitions and all public input tensors matched (40 complete games)`);
