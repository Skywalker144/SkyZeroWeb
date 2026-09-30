import {json,err} from '../_util.js';

export async function onRequestGet({env}){
 if(!env.POKER_ENGINE_URL||!env.POKER_ENGINE_TOKEN)return err('策略服务尚未配置',503);
 try{
  const response=await fetch(new URL('/health',env.POKER_ENGINE_URL),{headers:{Authorization:'Bearer '+env.POKER_ENGINE_TOKEN},signal:AbortSignal.timeout(10000)});
  if(!response.ok)return err('策略服务不可用',503);
  return json(await response.json());
 }catch{return err('策略服务不可用',503);}
}
