import {ApiError} from './_http.js';
export const REVISION='e9237e0c98114cadacd47a0f1b02f0d210b652f8';
const streetNames=['翻牌前','翻牌','转牌','河牌'];
export function toAuthorState(o){
 const street=streetNames.indexOf(o.street),bb=o.big_blind;
 if(street<0||!Number.isFinite(bb)||bb<=0||!Array.isArray(o.events))throw Error('原版策略需要结构化牌局信息，请刷新页面。');
 const convert=c=>{if(typeof c!=='string'||c.length!==2||!'23456789TJQKA'.includes(c[0]))throw Error('无效牌面');const s='♣♦♥♠'.indexOf(c[1]);if(s<0)throw Error('无效花色');return c[0]+'cdhs'[s];};
 const hole=o.your_cards.map(convert),board=o.board.map(convert);if(hole.length!==2||board.length!==[0,3,4,5][street]||new Set([...hole,...board]).size!==hole.length+board.length)throw Error('牌面数据不一致');
 const number=n=>{if(!Number.isFinite(n)||n<0)throw Error('无效筹码数据');return n/bb;};
 const events=[];let at=0;for(const e of o.events){if(!Number.isInteger(e.street)||e.street<at||e.street>street)throw Error('行动历史街次无效');while(at<e.street){events.push('_');at++;}const a={fold:'f',check:'k',call:'c',raise:'b'}[e.action];if(!a)throw Error('未知动作');events.push(a+(a==='b'?number(e.amount_to):''));}while(at<street){events.push('_');at++;}
 const l=o.legal,legal=[];if(l.fold)legal.push('f');if(l.check)legal.push('k');if(l.call>0)legal.push('c');if(l.raise)legal.push('b');
 const streetPot=o.pot-o.street_contributions.you-o.street_contributions.opponent;
 return {hand_id:o.hand,street:['preflop','flop','turn','river'][street],hero_hole_cards:hole.join(''),board_cards:board.join(''),pot:number(street===0?o.pot:streetPot),total_pot:number(o.pot),hero_stack:number(o.your_stack),villain_stack:number(o.opponent_stack),hero_position:o.position==='BTN/SB'?'BTN':'BB',legal_actions:legal,raise_min:l.raise?number(Math.min(l.min,l.max)):null,raise_max:l.raise?number(l.max):null,action_history:events,use_skills:true};
}

export async function buildAuthorStrategy(observation,env){
 if(!env.POKER_ENGINE_URL||!env.POKER_ENGINE_TOKEN)throw new ApiError(503,'策略服务尚未配置');
 let state;
 try{state=toAuthorState(observation);}catch{throw new ApiError(400,'牌局数据无效，请刷新页面后重试');}
 let response;
 try{response=await fetch(new URL('/analyze',env.POKER_ENGINE_URL).href,{method:'POST',headers:{Authorization:'Bearer '+env.POKER_ENGINE_TOKEN,'Content-Type':'application/json'},body:JSON.stringify({state}),signal:AbortSignal.timeout(20000)});}catch{throw new ApiError(503,'作者策略服务暂时不可用，请稍后重试');}
 if(!response.ok){await response.body?.cancel();throw new ApiError(response.status===400?400:503,'作者策略分析失败，请稍后重试');}
 let result;
 try{result=await response.json();}catch{throw new ApiError(503,'作者策略响应无效');}
 if(typeof result.result?.system_prompt!=='string'||typeof result.result?.user_prompt!=='string')throw new ApiError(503,'作者策略响应无效');
 const contract='\n\nGUI EXECUTION CONSTRAINTS (authoritative over strategy sizing suggestions):\nAll amounts are BB, 1 BB = '+observation.big_blind+' GUI chips. b means TOTAL invested on this street, including your previous street contribution of '+observation.street_contributions.you/observation.big_blind+' BB. It is NOT an additional payment. Legal action codes: '+state.legal_actions.join(', ')+'. '+(observation.legal.raise?'For b, amount must be between '+state.raise_min+' and '+state.raise_max+' BB inclusive, in increments of '+1/observation.big_blind+' BB. Round percentage-pot suggestions to that increment within these bounds.':'Bet/raise is forbidden.')+' Do not copy a suggested strategy size if it violates these limits.';
 return {state,system:result.result.system_prompt,user:result.result.user_prompt+contract,summary:{engine:'author',version:REVISION,street:observation.street,analysisMs:result.analysisMs,prompt:result.result.user_prompt+contract,systemPrompt:result.result.system_prompt,bigBlind:observation.big_blind}};
}
