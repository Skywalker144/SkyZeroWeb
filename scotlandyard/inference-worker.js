const assetQuery = self.location.search || `?v=${Date.now()}`;
importScripts('../onnx-worker-runtime.js' + assetQuery);
SkyZeroOnnxRuntime.loadOnnxRuntime(self);
let session = null;
let model = null;
function tensor(spec, values) {
  const type = {'tensor(int64)':'int64', 'tensor(float)':'float32', 'tensor(bool)':'bool'}[spec.type];
  if (!type) throw new Error(`Unsupported input type: ${spec.type}`);
  const data = type === 'int64' ? BigInt64Array.from(values, BigInt)
    : type === 'bool' ? Uint8Array.from(values) : Float32Array.from(values);
  return new ort.Tensor(type, data, spec.dims);
}
self.onmessage = async ({data}) => {
  try {
    if (data.kind === 'load') {
      const response = await fetch(data.url);
      if (!response.ok) throw new Error(`权重下载失败（HTTP ${response.status}）`);
      const progress = (stage, loaded) => self.postMessage({id:data.id,progress:{stage,loaded,total:data.model.bytes}});
      const reader = response.body?.getReader();
      let bytes;
      if (reader) {
        const chunks = [];
        let loaded = 0;
        while (true) {
          const {done,value} = await reader.read();
          if (done) break;
          loaded += value.byteLength;
          if (loaded > data.model.bytes) { await reader.cancel(); throw new Error('权重文件大小不符'); }
          chunks.push(value);
          progress('download', loaded);
        }
        const joined = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) { joined.set(chunk,offset); offset += chunk.byteLength; }
        bytes = joined.buffer;
      } else {
        bytes = await response.arrayBuffer();
        progress('download', bytes.byteLength);
      }
      progress('verify', bytes.byteLength);
      if (bytes.byteLength !== data.model.bytes) throw new Error('权重文件大小不符，请重新下载');
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2,'0')).join('');
      if (hash !== data.model.sha256) throw new Error('权重校验失败，请清除缓存后重试');
      progress('initialize', bytes.byteLength);
      const next = await ort.InferenceSession.create(bytes, {executionProviders:['wasm'], graphOptimizationLevel:'all'});
      if (session) await session.release();
      session = next;
      model = data.model;
      self.postMessage({id:data.id, result:true});
    } else if (data.kind === 'run') {
      if (!session) throw new Error('请先加载权重');
      const feeds = Object.fromEntries(model.inputs.map(spec => [spec.name,tensor(spec,data.inputs[spec.name])]));
      const result = await session.run(feeds);
      self.postMessage({id:data.id,result:{actions:Array.from(result.actions.data,Number),order:Array.from(result.order.data,Number)}});
    }
  } catch (error) {
    self.postMessage({id:data.id,error:String(error.message || error)});
  }
};
