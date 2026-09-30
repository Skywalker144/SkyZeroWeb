import {test} from 'node:test';import assert from 'node:assert/strict';import vm from 'node:vm';import {readFile} from 'node:fs/promises';
const html=await readFile(new URL('../public/poker/index.html',import.meta.url),'utf8'),script=html.split('<script type="module">')[1].split('</script>')[0];
function setup(reply,timers={}){const nodes={};const document={getElementById(id){return nodes[id]??=( {value:id==='key'?'fake-test-key':id==='effort'?'high':'',textContent:'',innerHTML:'',classList:{toggle(){}},focus(){}});},querySelectorAll(){return []}};const ctx=vm.createContext({document,crypto,performance,console,AbortController,TextDecoder,fetch:reply,TypeError,setTimeout:timers.setTimeout||setTimeout,clearTimeout,queueMicrotask,setInterval(){},confirm(){return true}});document.getElementById('key');vm.runInContext(script,ctx);return {ctx,nodes};}
function response(action,amount){const entries=[{choices:[{delta:{reasoning_content:'test'}}]},{choices:[{delta:{content:JSON.stringify({action,amount,reason:'测试说明'})},finish_reason:'stop'}]},{choices:[],usage:{prompt_tokens:100,completion_tokens:20,total_tokens:120}}];return new Response(entries.map(e=>'data: '+JSON.stringify(e)+'\n\n').join('')+'data: [DONE]\n\n');}
async function settle(ctx){for(let i=0;i<20;i++){await new Promise(r=>setTimeout(r,5));if(!vm.runInContext('busy',ctx))return;}throw Error('request not settled');}
test('streamed AI decisions advance twice across preflop and accumulate exact usage',async()=>{let count=0;const {ctx,nodes}=setup(async()=>{count++;return response('check');});vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.equal(count,2);assert.equal(vm.runInContext('game.street',ctx),1);assert.equal(vm.runInContext('game.actor',ctx),0);assert.match(nodes.handstats.textContent,/240 token/);assert.equal(nodes.tokens.textContent,120);assert.equal(vm.runInContext('error',ctx),'');});
test('illegal model action keeps game unchanged and enables retry',async()=>{const {ctx,nodes}=setup(async()=>response('raise',3));vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.equal(vm.runInContext('game.actor',ctx),1);assert.equal(vm.runInContext('game.street',ctx),0);assert.match(nodes.error.textContent,/不合法/);assert.equal(vm.runInContext('records.length',ctx),1);});
test('missing usage is marked unknown instead of invented tokens',async()=>{const {ctx,nodes}=setup(async()=>new Response('data: {"choices":[{"delta":{"content":"{\\"action\\":\\"fold\\"}"}}]}\n\ndata: [DONE]\n\n'));vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.equal(vm.runInContext('game.done',ctx),true);assert.match(nodes.handstats.textContent,/未知用量/);assert.equal(nodes.tokens.textContent,'—');});

test('original author protocol converts BB, carries hand history and resets at next hand',async()=>{
 const requests=[];const {ctx,nodes}=setup(async(url,opts)=>{const body=JSON.parse(opts.body);requests.push(body);const strategy={engine:'author',version:'test',bigBlind:2,analysisMs:5,prompt:'Original private prompt',street:body.observation.street};const action=requests.length===1?'b':'k';return new Response('data: '+JSON.stringify({pokerskill:strategy})+'\n\n'+await response(action,2.5).text());});
 nodes.skills.checked=true;nodes.stackmode.value='fixed200';vm.runInContext('start();move("call")',ctx);await settle(ctx);
 assert.equal(vm.runInContext('game.bets[1]',ctx),5);assert.equal(vm.runInContext('authorHistory.length',ctx),2);assert.equal(requests[0].observation.your_stack,398);assert(!nodes.review.innerHTML.includes('Original private prompt'));
 vm.runInContext('move("call")',ctx);await settle(ctx);assert.equal(requests[1].authorHistory.length,2);assert.equal(vm.runInContext('authorHistory.length',ctx),4);vm.runInContext('move("fold")',ctx);assert(nodes.review.innerHTML.includes('Original private prompt'));
 vm.runInContext('start()',ctx);await settle(ctx);assert.equal(requests.at(-1).authorHistory.length,0);
});

test('fractional author BB sizing rounds to a chip and records executed history',async()=>{
 const {ctx,nodes}=setup(async()=>new Response('data: '+JSON.stringify({pokerskill:{engine:'author',bigBlind:2,analysisMs:1,prompt:'test',version:'test'}})+'\n\n'+await response('b',2.65).text()));nodes.skills.checked=true;vm.runInContext('start();move("call")',ctx);await settle(ctx);
 assert.equal(vm.runInContext('game.bets[1]',ctx),5);assert.equal(vm.runInContext('error',ctx),'');assert.equal(vm.runInContext('JSON.parse(authorHistory[1].content).amount',ctx),2.5);assert.match(vm.runInContext('explanations[0]',ctx),/精度调整/);
});
test('out-of-range author sizing is rejected with limits and feedback on retry',async()=>{
 const requests=[];const {ctx,nodes}=setup(async(url,opts)=>{requests.push(JSON.parse(opts.body));return new Response('data: '+JSON.stringify({pokerskill:{engine:'author',bigBlind:2,analysisMs:1,prompt:'test',version:'test'}})+'\n\n'+await response('b',requests.length===1?1:3).text());});nodes.skills.checked=true;vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.match(nodes.error.textContent,/1 BB/);assert.match(nodes.error.textContent,/4–400 筹码/);assert.equal(vm.runInContext('game.bets[1]',ctx),2);vm.runInContext('askAI()',ctx);await settle(ctx);assert.match(requests[1].retryFeedback,/不合法/);assert.equal(vm.runInContext('game.bets[1]',ctx),6);
});

test('continuous 200BB carries settled stacks into next hand and explicit reset restores 400',async()=>{
 const {ctx,nodes}=setup(async()=>response('fold'));nodes.stackmode.value='running200';vm.runInContext('start();move("fold")',ctx);assert.equal(vm.runInContext('game.stacks[0]',ctx),399);assert.equal(vm.runInContext('game.stacks[1]',ctx),401);
 vm.runInContext('start()',ctx);await settle(ctx);assert.equal(vm.runInContext('game.stacks[0]',ctx),400);assert.equal(vm.runInContext('game.stacks[1]',ctx),400);
 vm.runInContext('start();move("fold");start(true)',ctx);assert.equal(vm.runInContext('game.stacks[0]+game.bets[0]',ctx),400);assert.equal(vm.runInContext('game.stacks[1]+game.bets[1]',ctx),400);
});
test('continuous mode preserves bust result until explicit reset',()=>{
 const {ctx,nodes}=setup(async()=>response('fold'));nodes.stackmode.value='running200';vm.runInContext('start();move("fold");game.stacks=[0,800];start()',ctx);assert.match(nodes.error.textContent,/筹码已耗尽/);assert.equal(vm.runInContext('game.stacks[1]',ctx),800);vm.runInContext('start(true)',ctx);assert.equal(vm.runInContext('game.stacks[0]+game.bets[0]',ctx),400);
});

const fastRetry={setTimeout:(fn,ms)=>setTimeout(fn,ms<10000?0:ms)};
test('network failure and truncated stream retry unchanged state, apply only one final action',async()=>{
 const requests=[];const {ctx,nodes}=setup(async(url,opts)=>{requests.push(opts.body);if(requests.length===1)throw new TypeError('Failed to fetch');if(requests.length===2)return new Response('data: {"choices":[{"delta":{"content":"{\\"action\\":\\"fold\\"}"}}]}\n\n');return response('fold');},fastRetry);
 vm.runInContext('start();move("call");askAI()',ctx);await settle(ctx);
 assert.equal(requests.length,3);assert.equal(new Set(requests).size,1);assert.equal(vm.runInContext('game.done',ctx),true);assert.equal(vm.runInContext('explanations.length',ctx),1);assert.equal(vm.runInContext('records.length',ctx),3);assert.equal(nodes.error.textContent,'');assert.match(nodes.handstats.textContent,/未知用量/);
});
test('transient HTTP errors stop after three retries and preserve the turn',async()=>{
 let count=0;const {ctx,nodes}=setup(async()=>{count++;return new Response(JSON.stringify({error:'temporarily unavailable'}),{status:503});},fastRetry);
 vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.equal(count,4);assert.equal(vm.runInContext('game.actor',ctx),1);assert.equal(vm.runInContext('records.length',ctx),4);assert.match(nodes.error.textContent,/手动重试/);
});
test('invalid key is not retried',async()=>{
 let count=0;const {ctx}=setup(async()=>{count++;return new Response(JSON.stringify({error:'key invalid'}),{status:401});},fastRetry);vm.runInContext('start();move("call")',ctx);await settle(ctx);assert.equal(count,1);
});
test('starting requires a DeepSeek key and preserves the empty table',()=>{
 const {ctx,nodes}=setup(async()=>response('fold'));nodes.key.value='';nodes.newhand.onclick();assert.equal(vm.runInContext('game',ctx),null);assert.match(nodes.error.textContent,/DeepSeek API key/);assert(!nodes.error.textContent.includes('陪练'));
});
test('history renders newest first without changing the observation order',()=>{
 const {ctx,nodes}=setup(async()=>response('fold'));vm.runInContext('start();move("fold")',ctx);assert(nodes.history.innerHTML.indexOf('赢得底池')<nodes.history.innerHTML.indexOf('弃牌</span>'));assert.match(vm.runInContext('game.observation().history[0].text',ctx),/小盲/);assert.equal(nodes.history.scrollTop,0);
});
test('raise input rejects invalid amounts and previews the additional contribution',()=>{
 const {ctx,nodes}=setup(async()=>response('fold'));vm.runInContext('start()',ctx);assert.equal(Number(nodes.amount.value),4);assert.match(nodes['raise-preview'].textContent,/再投入 3/);nodes.amount.value='3';nodes.amount.oninput();assert.equal(nodes.raise.disabled,true);nodes.amount.value='6';nodes.amount.oninput();assert.equal(nodes.raise.disabled,false);assert.match(nodes['raise-preview'].textContent,/再投入 5/);
});
test('cancel during backoff prevents the next request',async()=>{
 let count=0;const {ctx,nodes}=setup(async()=>{count++;throw new TypeError('offline');});vm.runInContext('start();move("call")',ctx);
 await new Promise(r=>setTimeout(r,10));assert.match(nodes.requeststate.textContent,/自动重试/);nodes.cancel.onclick();await settle(ctx);assert.equal(count,1);assert.equal(vm.runInContext('game.actor',ctx),1);assert.match(nodes.error.textContent,/已取消/);
});
