export class ApiError extends Error{
 constructor(status,message){super(message);this.status=status;}
}

export async function readJsonRequest(request){
 const origin=request.headers.get('origin');
 if(origin&&origin!==new URL(request.url).origin)throw new ApiError(403,'只接受本站页面请求');
 if(!request.headers.get('content-type')?.startsWith('application/json'))throw new ApiError(415,'需要 JSON 请求');
 if(!request.body)throw new ApiError(400,'请求参数无效');
 const reader=request.body.getReader(),decoder=new TextDecoder();let length=0,body='';
 try{
  while(true){const {value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>2000000){await reader.cancel();throw new ApiError(413,'请求过大');}body+=decoder.decode(value,{stream:true});}
  body+=decoder.decode();const parsed=JSON.parse(body);
  if(!parsed||typeof parsed!=='object'||Array.isArray(parsed))throw new ApiError(400,'请求参数无效');
  return parsed;
 }catch(error){if(error instanceof ApiError)throw error;throw new ApiError(400,'请求不是有效 JSON');}
 finally{reader.releaseLock();}
}
