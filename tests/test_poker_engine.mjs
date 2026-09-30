import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {toAuthorState} from '../functions/api/poker/_strategy.js';

test('real CPython 3.9 author engine authenticates, validates and isolates concurrent analyses',async()=>{
 const token='engine-test-token-'.repeat(4);
 const child=spawn(process.env.POKER_PYTHON||'python3.9',['-u','services/poker/engine.py'],{env:{...process.env,POKER_ENGINE_TOKEN:token,POKER_ENGINE_PORT:'0'},stdio:['ignore','pipe','pipe']});
 let stderr='';child.stderr.on('data',c=>stderr+=c);
 try{
  const ready=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error(stderr||'engine startup timed out')),15000);child.once('error',e=>{clearTimeout(timer);reject(e);});child.once('exit',()=>{clearTimeout(timer);reject(Error(stderr||'engine exited'));});child.stdout.once('data',c=>{clearTimeout(timer);try{resolve(JSON.parse(c.toString().trim()));}catch(e){reject(e);}});});
  assert.equal(ready.python.split('.').slice(0,2).join('.'),'3.9');const base=`http://127.0.0.1:${ready.port}`;
  assert.equal((await fetch(base+'/health')).status,401);
  const headers={'content-type':'application/json',authorization:'Bearer '+token};
  const health=await fetch(base+'/health',{headers});assert.equal(health.status,200);assert.equal((await health.json()).revision.length,40);
  assert.equal((await fetch(base+'/analyze',{method:'POST',headers,body:'not-json'})).status,400);
  assert.equal((await fetch(base+'/analyze',{method:'POST',headers,body:JSON.stringify({state:{street:'bad'}})})).status,400);
  for(const name of ['preflop_btn_open','flop_cbet','river_bluffcatch']){
   const state=JSON.parse(await readFile(`services/poker/vendor/pokerskill_agent/examples/${name}.json`,'utf8'));
   const deck=[...'23456789TJQKA'].flatMap(rank=>[...'cdhs'].map(suit=>rank+suit)).filter(card=>!state.board_cards?.includes(card));
   const states=Array.from({length:8},(_,i)=>({...state,use_skills:true,hand_id:i+1,hero_hole_cards:deck[i*2]+deck[i*2+1]}));
   const replies=await Promise.all(states.map(async(state)=>{const response=await fetch(base+'/analyze',{method:'POST',headers,body:JSON.stringify({state})});assert.equal(response.status,200);return response.json();}));
   for(const reply of replies){assert.match(reply.result.system_prompt,/ATTACK BUDGET/);assert(reply.result.user_prompt.length>300);assert(reply.analysisMs>=0);}
   for(let i=0;i<replies.length;i++){
    const response=await fetch(base+'/analyze',{method:'POST',headers,body:JSON.stringify({state:states[i]})});
    assert.equal(replies[i].result.user_prompt,(await response.json()).result.user_prompt);
   }
  }
  const html=await readFile('public/poker/index.html','utf8');const source=html.split('<script type="module">')[1].split('\nconst $=')[0];
  const {Game}=await import('data:text/javascript;base64,'+Buffer.from(source+'\nexport {Game};').toString('base64'));
  const streets=new Set(),positions=new Set();
  for(let h=0;h<2;h++){
   const game=new Game([400,400],h,h+1);
   while(!game.done){const state=toAuthorState(game.observation());const response=await fetch(base+'/analyze',{method:'POST',headers,body:JSON.stringify({state})});assert.equal(response.status,200);assert((await response.json()).result.user_prompt.length>300);streets.add(state.street);positions.add(state.hero_position);const legal=game.legal();game.act(legal.check?'check':'call');}
  }
  assert.equal(streets.size,4);assert.equal(positions.size,2);
  assert.equal((await fetch(base+'/analyze',{method:'POST',headers,body:'x'.repeat(70000)})).status,413);
 }finally{child.kill();await new Promise(resolve=>{if(child.exitCode!==null||child.signalCode!==null)resolve();else child.once('exit',resolve);});}
});
