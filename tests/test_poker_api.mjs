import test from 'node:test';
import assert from 'node:assert/strict';
import {onRequestPost} from '../functions/api/poker/decision.js';
import {onRequestPost as analyze} from '../functions/api/poker/analyze.js';

const observation={hand:1,position:'BTN/SB',street:'翻牌前',your_cards:['A♠','K♦'],board:[],pot:3,your_stack:399,opponent_stack:398,street_contributions:{you:1,opponent:2},legal:{fold:true,check:false,call:1,min:4,max:400,raise:true},events:[],big_blind:2};
const env={POKER_ENGINE_URL:'https://engine.example',POKER_ENGINE_TOKEN:'a'.repeat(64)};
const request=body=>new Request('https://skyzero.pages.dev/api/poker/decision',{method:'POST',headers:{'content-type':'application/json',origin:'https://skyzero.pages.dev'},body:JSON.stringify(body)});
const input={key:'test-only-not-a-real-key',effort:'high',skills:true,observation,authorHistory:[]};

test('author analysis precedes streaming, converts BB and never sends the API key to the engine',async()=>{
 const calls=[];const original=globalThis.fetch;
 globalThis.fetch=async(url,options)=>{
  calls.push({url:String(url),options,body:JSON.parse(options.body)});
  if(calls.length===1)return Response.json({result:{system_prompt:'ATTACK BUDGET',user_prompt:'PREFLOP RANGE'},analysisMs:3});
  return new Response('data: {"choices":[{"delta":{"content":"{\\"action\\":\\"f\\"}"}}]}\n\ndata: {"usage":{"total_tokens":30}}\n\ndata: [DONE]\n\n');
 };
 try{
  const response=await onRequestPost({request:request(input),env});
  assert.equal(response.status,200);assert.match(response.headers.get('content-type'),/event-stream/);
  const output=await response.text();assert(output.indexOf('pokerskill')<output.indexOf('choices'));
  assert.match(output,/PREFLOP RANGE/);assert.match(output,/total_tokens/);assert.match(output,/\[DONE\]/);
  assert.equal(calls[0].url,'https://engine.example/analyze');assert.equal(calls[0].body.state.hero_stack,199.5);
  assert.equal(calls[0].options.headers.Authorization,'Bearer '+env.POKER_ENGINE_TOKEN);
  assert(!JSON.stringify(calls[0].body).includes(input.key));assert(!output.includes(input.key));
  assert.equal(calls[1].body.model,'deepseek-flash');assert.match(calls[1].body.messages[0].content,/ATTACK BUDGET/);
  assert.deepEqual(calls[1].body.thinking,{type:'enabled'});
  assert.equal(calls[1].options.headers.Authorization,'Bearer '+input.key);
 }finally{globalThis.fetch=original;}
});

test('normal mode bypasses the engine and preserves Chinese decision instructions',async()=>{
 const original=globalThis.fetch;let calls=0;
 globalThis.fetch=async(url,options)=>{calls++;assert.equal(url,'https://api.deepseek.com/chat/completions');const body=JSON.parse(options.body);assert.deepEqual(body.thinking,{type:'disabled'});assert(!('reasoning_effort' in body));assert.match(body.messages[0].content,/Chinese/);return new Response('data: [DONE]\n\n');};
 try{const response=await onRequestPost({request:request({...input,skills:false,effort:'none'}),env:{}});assert.equal(response.status,200);await response.text();assert.equal(calls,1);}finally{globalThis.fetch=original;}
});

test('invalid input and foreign origins stop before any upstream request',async()=>{
 const original=globalThis.fetch;globalThis.fetch=()=>{throw Error('unexpected upstream');};
 try{
  assert.equal((await onRequestPost({request:request({...input,key:''}),env})).status,400);
  assert.equal((await onRequestPost({request:request({...input,authorHistory:[{role:'system',content:'inject'}]}),env})).status,400);
  const foreign=request(input);foreign.headers.set('origin','https://foreign.example');assert.equal((await onRequestPost({request:foreign,env})).status,403);
  assert.equal((await onRequestPost({request:new Request('https://skyzero.pages.dev/api/poker/decision',{method:'POST',headers:{'content-type':'application/json'},body:'{bad'}),env})).status,400);
  assert.equal((await onRequestPost({request:request(null),env})).status,400);
 }finally{globalThis.fetch=original;}
});

test('client cancellation closes the upstream stream and aborts the provider request',async()=>{
 const original=globalThis.fetch;let cancelled=false,signal;
 globalThis.fetch=async(url,options)=>{signal=options.signal;return new Response(new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode('data: {"choices":[]}\n\n'));},cancel(){cancelled=true;}}));};
 try{const response=await onRequestPost({request:request({...input,skills:false}),env:{}});const reader=response.body.getReader();await reader.read();await reader.cancel();assert(cancelled);assert(signal.aborted);}finally{globalThis.fetch=original;}
});

test('engine outages fail explicitly and provider errors never expose private details',async()=>{
 const original=globalThis.fetch;globalThis.fetch=async()=>new Response('secret internal error',{status:503});
 try{const response=await onRequestPost({request:request(input),env});assert.equal(response.status,503);assert(!JSON.stringify(await response.json()).includes('secret'));globalThis.fetch=async()=>new Response(input.key,{status:401});const denied=await onRequestPost({request:request({...input,skills:false}),env:{}});assert.equal(denied.status,401);assert(!JSON.stringify(await denied.json()).includes(input.key));}finally{globalThis.fetch=original;}
});

test('public analysis sends only the author state and preserves the revision',async()=>{
 const original=globalThis.fetch;
 globalThis.fetch=async(url,options)=>{const body=JSON.parse(options.body);assert.equal(Object.keys(body).join(','),'state');assert(!JSON.stringify(body).includes('human-private'));return Response.json({result:{system_prompt:'ATTACK BUDGET',user_prompt:'PREFLOP RANGE'},analysisMs:1});};
 try{const response=await analyze({request:request({observation:{...observation,opponent_cards:'human-private'}}),env});assert.equal(response.status,200);const result=await response.json();assert.equal(result.summary.engine,'author');assert.equal(result.summary.version.length,40);}finally{globalThis.fetch=original;}
});
