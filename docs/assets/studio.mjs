import {BDD_LABELS,COCO_LABELS,exportLabel,visibleBoxes,getReplay,createRequestGate,toCvatXml} from './core.mjs';
const $=id=>document.getElementById(id);
const canvas=$('canvas'),ctx=canvas.getContext('2d'),gate=createRequestGate();
const state={manifest:null,name:'BDD01.jpg',image:null,hash:null,isUpload:false,boxes:[],original:[],selected:null,
  result:null,tool:'select',history:[],zoom:1,panX:0,panY:0,space:false,ready:false,busy:false};
const cache=new Map(),reviewed=new Map();let worker,sequence=0,uploaded=null,drag=null;
const clone=value=>structuredClone(value);
const key=()=>`${state.hash}|${$('runtime').value}`;
const conf=()=>Number($('confidence').value);
const shown=()=>visibleBoxes(state.boxes,conf(),$('scope').value);
const status=(text,kind='')=>{$('status').textContent=text;$('status').dataset.kind=kind;};
const sha=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
function checkpoint(){state.history.push(clone(state.boxes));if(state.history.length>50)state.history.shift();}
function invalidateReview(){reviewed.delete(state.name);$('review-status').textContent='Chưa kiểm';$('review-status').classList.remove('reviewed');updateActions();}
function persist(){if(state.ready&&state.hash)cache.set(key(),{boxes:clone(state.boxes),original:clone(state.original),result:clone(state.result),history:clone(state.history)});}
function record(){return {name:state.name,width:state.image.naturalWidth,height:state.image.naturalHeight,
  image_sha256:state.hash,runtime:$('runtime').value,confidence:conf(),scope:$('scope').value,boxes:clone(shown()),result:state.result};}
function updateActions(){
  $('run').disabled=state.busy||!state.image;
  $('export').disabled=!state.ready; $('review').disabled=!state.ready;
  $('export-reviewed').disabled=!reviewed.size;
  $('export-session').disabled=!state.ready&&!reviewed.size;
  $('review-count').textContent=`${reviewed.size} ảnh đã kiểm trong phiên. Thay đổi nhãn hoặc bộ lọc sẽ hủy dấu đã kiểm.`;
  $('undo').disabled=!state.history.length||!state.ready;
}
function refreshList(){
  const boxes=shown();if(!boxes.some(b=>b.id===state.selected))state.selected=null;
  $('box-count').textContent=String(boxes.length);$('box-list').replaceChildren();
  if(!boxes.length){const p=document.createElement('p');p.className='empty';p.textContent=state.ready?'Không có hộp vượt bộ lọc. Có thể giảm confidence hoặc vẽ bổ sung.':'Chưa có prediction cho ảnh và nguồn này. Bấm Chạy prelabel.';$('box-list').append(p);}
  boxes.forEach((b,i)=>{const row=document.createElement('button');row.className=`box-row ${b.id===state.selected?'selected':''}`;
    row.setAttribute('aria-pressed',String(b.id===state.selected));
    const name=document.createElement('span');name.textContent=`${i+1}. ${exportLabel(b,$('scope').value)}`;
    const score=document.createElement('span');score.className='score';score.textContent=b.edited?'đã sửa':`${(100*b.confidence).toFixed(0)}%`;
    row.append(name,score);row.onclick=()=>{state.selected=b.id;refreshList();draw();};$('box-list').append(row);});
  const selected=state.boxes.find(b=>b.id===state.selected);$('editor').hidden=!selected;
  if(selected){$('label').value=exportLabel(selected,$('scope').value);['x1','y1','x2','y2'].forEach((id,i)=>{$(id).value=selected.box_xyxy[i].toFixed(1);});}
  updateActions();
}
function transform(){const rect=canvas.getBoundingClientRect();const w=state.image?.naturalWidth||1,h=state.image?.naturalHeight||1;
  const scale=Math.min(rect.width/w,rect.height/h)*state.zoom;
  return {scale,x:(rect.width-w*scale)/2+state.panX,y:(rect.height-h*scale)/2+state.panY};}
function draw(){
  const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);if(!state.image)return;
  const {scale,x,y}=transform();ctx.drawImage(state.image,x,y,state.image.naturalWidth*scale,state.image.naturalHeight*scale);
  for(const b of shown()){
    const [a,c,d,e]=b.box_xyxy,selected=b.id===state.selected;
    ctx.strokeStyle=selected?'#ffbd69':'#60f0ae';ctx.lineWidth=selected?2.5:1.5;
    ctx.strokeRect(x+a*scale,y+c*scale,(d-a)*scale,(e-c)*scale);
    const label=`${exportLabel(b,$('scope').value)} ${b.edited?'*':(100*b.confidence).toFixed(0)+'%'}`;
    ctx.font='600 11px Segoe UI,sans-serif';const tw=ctx.measureText(label).width;
    const lx=x+a*scale,ly=Math.max(0,y+c*scale-18);ctx.fillStyle=ctx.strokeStyle;ctx.fillRect(lx,ly,tw+9,18);ctx.fillStyle='#132326';ctx.fillText(label,lx+4,ly+13);
    if(selected){ctx.fillStyle='#ffbd69';ctx.fillRect(x+d*scale-5,y+e*scale-5,10,10);}
  }
  $('zoom-level').textContent=state.zoom===1?'Fit':`${Math.round(state.zoom*100)}%`;
}
function fit(){state.zoom=1;state.panX=state.panY=0;draw();}
function selectTool(tool){state.tool=tool;for(const name of ['select','draw']){$(`${name}-tool`).classList.toggle('active',tool===name);$(`${name}-tool`).setAttribute('aria-pressed',String(tool===name));}}
function clearResult(){state.ready=false;state.boxes=[];state.result=null;state.selected=null;state.history=[];
  $('result-source').textContent='Chưa có kết quả suy luận';$('timing').textContent='Thời gian: —';refreshList();draw();}
function applyResult(boxes,result){
  state.boxes=boxes.map(b=>({...b,id:++sequence}));state.original=clone(state.boxes);state.result=result;state.ready=true;state.history=[];
  $('result-source').textContent=result.source;
  $('timing').textContent=result.latency_ms==null?'Thời gian: —':`${result.timing_label}: ${result.latency_ms.toFixed(1)} ms`;
  $('provenance').textContent=JSON.stringify({image:state.name,image_sha256:state.hash,...result,notice:'Predictions cần review; chưa có mAP GT độc lập.'},null,2);
  refreshList();draw();persist();
}
async function loadCurrent(){
  const id=gate.next();clearResult();state.image=null;updateActions();$('overlay').textContent='Đang nạp ảnh…';
  $('image-name').textContent=state.name;$('image-size').textContent='';
  const review=reviewed.get(state.name);
  const isReviewed=review?.runtime===$('runtime').value&&review?.confidence===conf()&&review?.scope===$('scope').value;
  $('review-status').textContent=isReviewed?'Đã kiểm':'Chưa kiểm';
  $('review-status').classList.toggle('reviewed',isReviewed);
  document.querySelectorAll('.sample').forEach(b=>{b.classList.toggle('active',b.dataset.name===state.name);b.setAttribute('aria-pressed',String(b.dataset.name===state.name));});
  try{
    const file=state.isUpload?uploaded:null;
    const response=file?null:await fetch(new URL(`../images/${encodeURIComponent(state.name)}`,import.meta.url));
    if(response&&!response.ok)throw new Error('Không tải được ảnh mẫu.');
    const bytes=file?await file.arrayBuffer():await response.arrayBuffer();const hash=await sha(bytes);
    const expected=state.manifest.images[state.name]?.sha256;
    if(!file&&hash!==expected)throw new Error('Hash ảnh không khớp manifest; dừng để tránh gán nhãn sai ảnh.');
    const blob=new Blob([bytes],{type:file?.type||'image/jpeg'}),objectUrl=URL.createObjectURL(blob);
    const image=new Image();try{await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Ảnh không đọc được.'));image.src=objectUrl;});}finally{URL.revokeObjectURL(objectUrl);}
    if(!gate.current(id))return;state.image=image;state.hash=hash;$('overlay').textContent='';
    $('image-size').textContent=`${image.naturalWidth} × ${image.naturalHeight}`;fit();
    const cached=cache.get(key());
    if(cached){applyResult(cached.boxes,cached.result);state.boxes=cached.boxes;state.original=cached.original;state.history=cached.history;refreshList();draw();status('Đã khôi phục nhãn trong phiên cho đúng ảnh và nguồn.','success');}
    else if($('runtime').value.startsWith('replay:')&&!file)await runReplay();
    else {status('Ảnh sẵn sàng. Bấm Chạy prelabel để suy luận trên ảnh này.');$('provenance').textContent=JSON.stringify({image:state.name,image_sha256:state.hash,quality:state.manifest.quality},null,2);}
    updateActions();
  }catch(error){if(gate.current(id)){$('overlay').textContent='Không thể nạp ảnh';status(error.message,'error');updateActions();}}
}
async function runReplay(){
  const runtime=$('runtime').value.split(':')[1];const {record,result}=getReplay(state.manifest,state.name,runtime);
  if(record.sha256!==state.hash)throw new Error('Prediction không thuộc ảnh đang xem.');
  applyResult(clone(result.detections),{source:`Kết quả đã chạy · ${runtime}`,model_sha256:state.manifest.models[runtime].sha256,
    latency_ms:result.latency_ms,timing_label:'Đã đo trên máy tạo dữ liệu',generated_at:state.manifest.generated_at,
    hardware:state.manifest.hardware,preprocessing:state.manifest.preprocessing,model_quality:state.manifest.models[runtime].quality_note});
  status('Đã nạp kết quả inference lưu sẵn của đúng ảnh. Thời gian bên trên thuộc máy tạo dữ liệu, không phải browser này.','success');
}
function prepareTensor(){
  const image=state.image,size=640,ratio=Math.min(size/image.naturalWidth,size/image.naturalHeight);
  const rw=Math.round(image.naturalWidth*ratio),rh=Math.round(image.naturalHeight*ratio);
  const left=Math.round((size-rw)/2-.1),top=Math.round((size-rh)/2-.1);
  const off=document.createElement('canvas');off.width=off.height=size;const c=off.getContext('2d',{willReadFrequently:true});
  c.fillStyle='rgb(114,114,114)';c.fillRect(0,0,size,size);c.drawImage(image,left,top,rw,rh);
  const rgba=c.getImageData(0,0,size,size).data,plane=size*size,tensor=new Float32Array(3*plane);
  for(let i=0;i<plane;i++){tensor[i]=rgba[4*i]/255;tensor[plane+i]=rgba[4*i+1]/255;tensor[2*plane+i]=rgba[4*i+2]/255;}
  return {tensor,geometry:{width:image.naturalWidth,height:image.naturalHeight,ratio,left,top}};
}
function browserPredict(id){
  if(!worker)worker=new Worker(new URL('./inference-worker.js',import.meta.url));
  const model=state.manifest.models.onnx,{tensor,geometry}=prepareTensor();
  return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>{worker.terminate();worker=null;reject(new Error('Inference quá 3 phút. Thử lại hoặc chọn kết quả đã chạy.'));},180000);
    worker.onerror=()=>{clearTimeout(timeout);worker?.terminate();worker=null;reject(new Error('Browser không khởi tạo được ONNX/WASM. Có thể chọn kết quả đã chạy.'));};
    worker.onmessage=({data})=>{if(data.id!==id)return;
      if(data.type==='progress'){if(gate.current(id))status(data.text,'busy');return;}
      clearTimeout(timeout);if(data.type==='error')reject(new Error(data.text));else resolve(data);
    };
    worker.postMessage({id,modelUrl:new URL('../models/yolov8s.onnx',import.meta.url).href,modelHash:model.sha256,tensor,geometry},[tensor.buffer]);
  });
}
async function run(){
  if(!state.image||state.busy)return;
  if(state.ready&&state.boxes.some(b=>b.edited)&&!confirm('Chạy lại sẽ thay nhãn đã sửa của ảnh này. Tiếp tục?'))return;
  const id=gate.next(),runtime=$('runtime').value,imageName=state.name;
  state.busy=true;clearResult();invalidateReview();updateActions();status('Đang chuẩn bị suy luận…','busy');
  try{
    if(runtime.startsWith('replay:'))await runReplay();
    else if(runtime==='browser'){
      const result=await browserPredict(id);if(!gate.current(id))return;
      applyResult(result.boxes,{source:'Chạy thật · ONNX Browser CPU (WASM, 1 thread)',
        model_sha256:state.manifest.models.onnx.sha256,latency_ms:result.inference_ms,timing_label:'Inference browser, chưa tính tải model',
        preprocessing:'RGB letterbox 640; NMS theo class; COCO pretrained'});
      status('Inference hoàn tất trên browser. Kiểm tra hộp, bổ sung vật thể bỏ sót rồi xuất nhãn.','success');
    }else{
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_name:imageName,runtime:runtime.split(':')[1],conf:.1})});
      const data=await response.json();if(!response.ok)throw new Error(data.error||'Inference API lỗi.');
      if(!gate.current(id))return;
      if(data.image_name!==imageName||data.image_sha256!==state.hash||data.width!==state.image.naturalWidth||data.height!==state.image.naturalHeight)throw new Error('API trả sai ảnh hoặc hash/kích thước không khớp.');
      applyResult(data.detections,{source:`Chạy thật · Local ${data.runtime}`,latency_ms:data.latency_ms,timing_label:'Inference CPU local'});
      status('CPU local đã suy luận xong; nhãn cần được kiểm tra.','success');
    }
  }catch(error){if(gate.current(id)){clearResult();status(error.message,'error');}}
  finally{state.busy=false;updateActions();}
}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportRecords(records){try{download('m38_annotations.xml',toCvatXml(records),'application/xml');status(`Đã xuất ${records.length} ảnh, ${records.reduce((n,r)=>n+r.boxes.length,0)} hộp đang hiển thị. Import bằng CVAT for images 1.1.`,'success');}catch(e){status(e.message,'error');}}
function selectBox(){return state.boxes.find(b=>b.id===state.selected);}
function applyEdit(){
  const b=selectBox();if(!b)return;const points=['x1','y1','x2','y2'].map(id=>Number($(id).value));
  const [a,c,d,e]=points;if(!points.every(Number.isFinite)||!(0<=a&&a<d&&d<=state.image.naturalWidth&&0<=c&&c<e&&e<=state.image.naturalHeight)){status('Tọa độ phải nằm trong ảnh, cạnh phải/dưới lớn hơn trái/trên.','error');return;}
  checkpoint();b.box_xyxy=points;b.class_name=$('label').value;b.edited=true;invalidateReview();refreshList();draw();persist();
}
function removeBox(){if(!selectBox())return;checkpoint();state.boxes=state.boxes.filter(b=>b.id!==state.selected);state.selected=null;invalidateReview();refreshList();draw();persist();}
function undo(){if(!state.history.length)return;state.boxes=state.history.pop();invalidateReview();refreshList();draw();persist();}
function point(event){const rect=canvas.getBoundingClientRect(),t=transform();return [Math.max(0,Math.min(state.image.naturalWidth,(event.clientX-rect.left-t.x)/t.scale)),Math.max(0,Math.min(state.image.naturalHeight,(event.clientY-rect.top-t.y)/t.scale))];}
canvas.onpointerdown=event=>{
  if(!state.image)return;canvas.focus();canvas.setPointerCapture(event.pointerId);
  if(state.space||event.button===1){drag={mode:'pan',client:[event.clientX,event.clientY],pan:[state.panX,state.panY]};return;}
  if(!state.ready)return;const p=point(event);
  if(state.tool==='draw'){
    checkpoint();const b={id:++sequence,class_name:$('label').value,class_id:-1,confidence:1,edited:true,box_xyxy:[...p,...p]};state.boxes.push(b);state.selected=b.id;drag={mode:'draw',start:p,id:b.id};
  }else{
    const tolerance=10/transform().scale;const selected=selectBox();
    if(selected&&Math.abs(selected.box_xyxy[2]-p[0])<tolerance&&Math.abs(selected.box_xyxy[3]-p[1])<tolerance){checkpoint();drag={mode:'resize',id:selected.id,origin:[...selected.box_xyxy]};}
    else{const hit=[...shown()].reverse().find(b=>p[0]>=b.box_xyxy[0]&&p[0]<=b.box_xyxy[2]&&p[1]>=b.box_xyxy[1]&&p[1]<=b.box_xyxy[3]);state.selected=hit?.id??null;if(hit){checkpoint();drag={mode:'move',id:hit.id,start:p,origin:[...hit.box_xyxy]};}}
  }refreshList();draw();
};
canvas.onpointermove=event=>{
  if(!drag)return;if(drag.mode==='pan'){state.panX=drag.pan[0]+event.clientX-drag.client[0];state.panY=drag.pan[1]+event.clientY-drag.client[1];draw();return;}
  const b=state.boxes.find(b=>b.id===drag.id);if(!b)return;const p=point(event),o=drag.origin;
  if(drag.mode==='draw')b.box_xyxy=[Math.min(p[0],drag.start[0]),Math.min(p[1],drag.start[1]),Math.max(p[0],drag.start[0]),Math.max(p[1],drag.start[1])];
  if(drag.mode==='resize')b.box_xyxy=[o[0],o[1],Math.max(o[0]+1,p[0]),Math.max(o[1]+1,p[1])];
  if(drag.mode==='move'){const dx=Math.max(-o[0],Math.min(state.image.naturalWidth-o[2],p[0]-drag.start[0])),dy=Math.max(-o[1],Math.min(state.image.naturalHeight-o[3],p[1]-drag.start[1]));b.box_xyxy=[o[0]+dx,o[1]+dy,o[2]+dx,o[3]+dy];}
  b.edited=true;draw();
};
function endDrag(){if(drag&&drag.mode!=='pan'){state.boxes=state.boxes.filter(b=>b.box_xyxy[2]-b.box_xyxy[0]>=1&&b.box_xyxy[3]-b.box_xyxy[1]>=1);invalidateReview();refreshList();persist();}drag=null;draw();}
canvas.onpointerup=endDrag;canvas.onpointercancel=endDrag;
canvas.addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.5,Math.min(8,state.zoom*(event.deltaY<0?1.12:1/1.12)));draw();},{passive:false});
document.addEventListener('keydown',event=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName))return;
  if(event.code==='Space'){event.preventDefault();state.space=true;}
  if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();removeBox();}
  if(event.key.toLowerCase()==='b')selectTool('draw');if(event.key.toLowerCase()==='v')selectTool('select');
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();undo();}
});
document.addEventListener('keyup',event=>{if(event.code==='Space')state.space=false;});window.addEventListener('blur',()=>{state.space=false;endDrag();});
$('run').onclick=run;$('select-tool').onclick=()=>selectTool('select');$('draw-tool').onclick=()=>selectTool('draw');$('fit').onclick=fit;
$('zoom-in').onclick=()=>{state.zoom=Math.min(8,state.zoom*1.25);draw();};$('zoom-out').onclick=()=>{state.zoom=Math.max(.5,state.zoom/1.25);draw();};$('undo').onclick=undo;
$('apply-box').onclick=applyEdit;$('delete-box').onclick=removeBox;
for(const id of ['confidence','scope'])$(id).oninput=()=>{$('confidence-value').textContent=conf().toFixed(2);invalidateReview();refreshList();draw();};
$('runtime').onchange=()=>{
  // Persist under the previous runtime, not the newly selected value.
  const previous=$('runtime').dataset.previous||'browser',current=$('runtime').value;
  $('runtime').value=previous;persist();$('runtime').value=current;$('runtime').dataset.previous=current;
  $('runtime-help').textContent=current.startsWith('replay:')?'Predictions đã chạy trên máy tạo dữ liệu; không phải suy luận live trong browser.':current==='browser'?'ONNX chạy trên CPU thiết bị của bạn. Lần đầu tải model ~43 MB; tốc độ tùy thiết bị.':'Inference thật qua server CPU local. Chỉ ảnh mẫu được gửi tên đến API.';
  if(state.isUpload&&current!=='browser'){state.isUpload=false;state.name=Object.keys(state.manifest.images)[0];}
  loadCurrent();
};
$('upload').onchange=async()=>{const file=$('upload').files[0];if(!file)return;if(file.size>25*1048576){status('Chọn ảnh nhỏ hơn 25 MB.','error');return;}persist();uploaded=file;state.isUpload=true;state.name=file.name;$('runtime').value='browser';$('runtime').dataset.previous='browser';await loadCurrent();};
$('review').onclick=()=>{if(!state.ready)return;reviewed.set(state.name,record());$('review-status').textContent='Đã kiểm';$('review-status').classList.add('reviewed');updateActions();status('Đã đánh dấu ảnh này được kiểm với bộ lọc hiện tại.','success');};
$('export').onclick=()=>{if(state.ready)exportRecords([record()]);};$('export-reviewed').onclick=()=>exportRecords([...reviewed.values()]);
$('export-session').onclick=()=>download('m38_session.json',JSON.stringify({schema:'m38-review-session-v1',current:state.ready?record():null,reviewed:[...reviewed.values()]},null,2),'application/json');
new ResizeObserver(draw).observe($('stage'));
async function init(){
  try{
    if(matchMedia('(max-width:760px)').matches)document.querySelector('.controls').open=false;
    const response=await fetch(new URL('../predictions/manifest.json',import.meta.url));if(!response.ok)throw new Error('Thiếu prediction manifest.');state.manifest=await response.json();
    const names=Object.keys(state.manifest.images);$('sample-count').textContent=String(names.length);
    for(const name of names){const b=document.createElement('button');b.textContent=name;b.dataset.name=name;b.className='sample';b.onclick=()=>{persist();state.name=name;state.isUpload=false;loadCurrent();};$('samples').append(b);}
    for(const name of [...new Set([...BDD_LABELS,...COCO_LABELS])]){const o=document.createElement('option');o.value=o.textContent=name;$('label').append(o);}$('label').value='car';
    if(['localhost','127.0.0.1','[::1]'].includes(location.hostname)){
      try{const caps=await fetch('/api/capabilities');if(caps.ok){const data=await caps.json();for(const runtime of data.runtimes){const o=document.createElement('option');o.value=`local:${runtime}`;o.textContent=`Chạy thật · CPU local ${runtime}`;$('runtime').append(o);}}}catch{}
    }
    state.name=names[0];await loadCurrent();
  }catch(error){status(error.message,'error');$('overlay').textContent='Không nạp được dữ liệu. Kiểm tra kết nối hoặc bản build.';}
}
init();
