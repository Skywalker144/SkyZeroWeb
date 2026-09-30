import {json,err} from '../_util.js';
import {ApiError,readJsonRequest} from './_http.js';
import {buildAuthorStrategy} from './_strategy.js';

const prompt=`You play heads-up no-limit Texas Hold'em against a human. You are an opponent, not the referee. You receive only your own private cards and public information. Choose exactly one legal action from the observation. Return JSON only: {"action":"fold"|"check"|"call"|"raise","amount":integer,"reason":"brief Chinese explanation for post-hand review"}. amount is required only for raise and is the TOTAL contribution on the current street, not the increment. legal.min is the minimum full raise-to, legal.max is your all-in raise-to. A raise below min is legal only if it equals max. Never raise when legal.raise is false. Call only if legal.call>0; check only if legal.check. Play thoughtfully with a mix of value bets and bluffs. Do not invent cards or information.`;

export async function onRequestPost({request,env}){
 const abort=new AbortController();let timer;
 try{
  const {key,effort,observation,skills=false,authorHistory=[],retryFeedback=''}=await readJsonRequest(request);
  if(typeof key!=='string'||!key.trim()||key.length>512)throw new ApiError(400,'请填写有效 API key');
  if(!['none','low','high','max'].includes(effort)||!observation||!Array.isArray(observation.your_cards)||typeof skills!=='boolean')throw new ApiError(400,'请求参数无效');
  if(!Array.isArray(authorHistory)||authorHistory.length>120||authorHistory.some(m=>!['user','assistant'].includes(m.role)||typeof m.content!=='string'||m.content.length>50000))throw new ApiError(400,'手内对话记录无效');
  const strategy=skills?await buildAuthorStrategy(observation,env):null;
  const feedback=typeof retryFeedback==='string'&&retryFeedback?`\nPrevious attempt was rejected without changing the game: ${retryFeedback.slice(0,1000)}. Correct the action and sizing using the current legal limits.`:'';
  const messages=strategy?[{role:'system',content:strategy.system+'\nRespond as JSON with action f/k/c/b, amount in BB as TOTAL contribution on this street, reasoning as a brief Chinese explanation. This adapter overrides the integer-BB output rule: amount must be a multiple of '+(1/observation.big_blind)+' BB (one chip). Do not output text outside JSON.'},...authorHistory,{role:'user',content:strategy.user+feedback}]:[{role:'system',content:prompt},{role:'user',content:JSON.stringify(observation)+feedback}];
  request.signal.addEventListener('abort',()=>abort.abort(),{once:true});
  timer=setTimeout(()=>abort.abort(),175000);
  const upstream=await fetch('https://api.deepseek.com/chat/completions',{method:'POST',headers:{Authorization:'Bearer '+key.trim(),'Content-Type':'application/json'},body:JSON.stringify({model:'deepseek-flash',messages,thinking:{type:effort==='none'?'disabled':'enabled'},...(effort==='none'?{}:{reasoning_effort:effort}),max_tokens:16384,response_format:{type:'json_object'},stream:true,stream_options:{include_usage:true}}),signal:abort.signal});
  if(!upstream.ok){const errors={400:'DeepSeek 拒绝请求参数，请检查模型可用性。',401:'API key 无效，请检查后重试。',402:'DeepSeek 余额不足。',429:'DeepSeek 请求限流，请稍后重试。'};await upstream.body?.cancel();throw new ApiError(upstream.status,errors[upstream.status]||'DeepSeek 服务暂时不可用');}
  if(!upstream.body)throw new ApiError(502,'DeepSeek 未返回流式响应');
  const reader=upstream.body.getReader(),encoder=new TextEncoder();let heartbeat;
  const stream=new ReadableStream({
   async start(controller){
    try{
     if(strategy)controller.enqueue(encoder.encode('data: '+JSON.stringify({pokerskill:strategy.summary})+'\n\n'));
     heartbeat=setInterval(()=>controller.enqueue(encoder.encode(': waiting\n\n')),10000);
     while(true){const {value,done}=await reader.read();if(done)break;controller.enqueue(value);}
     controller.close();
    }catch{try{controller.enqueue(encoder.encode('data: '+JSON.stringify({error:{message:'模型响应中断或超时，牌局保留，请重试。'}})+'\n\n'));controller.close();}catch{}}
    finally{clearTimeout(timer);clearInterval(heartbeat);reader.releaseLock();}
   },
   async cancel(){clearTimeout(timer);clearInterval(heartbeat);abort.abort();await reader.cancel().catch(()=>{});}
  });
  return new Response(stream,{headers:{'Content-Type':'text/event-stream; charset=utf-8','Cache-Control':'no-cache, no-store','X-Accel-Buffering':'no'}});
 }catch(error){clearTimeout(timer);abort.abort();return error instanceof ApiError?err(error.message,error.status):err('无法连接 DeepSeek，请检查网络后重试。',502);}
}
