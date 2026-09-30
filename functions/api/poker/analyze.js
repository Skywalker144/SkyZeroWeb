import {json,err} from '../_util.js';
import {readJsonRequest,ApiError} from './_http.js';
import {buildAuthorStrategy} from './_strategy.js';

export async function onRequestPost({request,env}){
 try{const {observation}=await readJsonRequest(request);return json(await buildAuthorStrategy(observation,env));}
 catch(error){return err(error instanceof ApiError?error.message:'策略分析失败',error instanceof ApiError?error.status:503);}
}
