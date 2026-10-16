/* ONNX CPU inference stays off the UI thread. No images leave the browser. */
importScripts('../vendor/ort.wasm.min.js');
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('../vendor/',self.location.href).href;
let sessionPromise;
async function loadModel(url,expectedHash,id){
  const response=await fetch(url);if(!response.ok)throw new Error('Không tải được model ONNX.');
  const length=Number(response.headers.get('content-length'));const reader=response.body.getReader();
  const chunks=[];let received=0;
  while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;
    self.postMessage({id,type:'progress',text:`Đang tải model: ${(received/1048576).toFixed(1)} MB${length?' / '+(length/1048576).toFixed(1)+' MB':''}`});}
  const bytes=new Uint8Array(received);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==expectedHash)throw new Error('Model hash không khớp manifest; dừng inference.');
  self.postMessage({id,type:'progress',text:'Đang khởi tạo ONNX Runtime CPU…'});
  return ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
}
self.onmessage=async({data})=>{
  const {id,modelUrl,modelHash,tensor,geometry}=data;
  try{
    if(!sessionPromise)sessionPromise=loadModel(modelUrl,modelHash,id).catch(e=>{sessionPromise=null;throw e;});
    const session=await sessionPromise;
    self.postMessage({id,type:'progress',text:'Đang suy luận trên CPU của browser…'});
    const start=performance.now();
    const output=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',tensor,[1,3,640,640])});
    const elapsed=performance.now()-start;
    const result=output[session.outputNames[0]];
    if(result.dims[1]!==84)throw new Error('Model output không phải YOLO COCO [1,84,N].');
    const {decodeYolo}=await import('./core.mjs');
    const boxes=decodeYolo(result.data,result.dims[2],geometry,.1,.45);
    self.postMessage({id,type:'result',boxes,inference_ms:elapsed});
  }catch(error){self.postMessage({id,type:'error',text:error.message});}
};
