export const BDD_LABELS=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign'];
export const COCO_LABELS=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
const TRAFFIC=new Set([0,1,2,3,5,6,7,9,11]);
export const labelFor=(name)=>name==='person'?'pedestrian':name==='stop sign'?'traffic sign':name;
export const exportLabel=(box,scope)=>scope==='all'||box.edited?box.class_name:labelFor(box.class_name);
export function visibleBoxes(boxes,confidence,scope){
  return boxes.filter(b=>(b.edited || b.confidence>=confidence) && (scope==='all'||b.edited||TRAFFIC.has(b.class_id)));
}
export function getReplay(manifest,image,runtime){
  const record=manifest.images?.[image]; const result=record?.runtimes?.[runtime];
  if(!result)throw new Error(`Không có prediction cho ${image} / ${runtime}.`);
  return {record,result};
}
export function createRequestGate(){let generation=0;return {next:()=>++generation,current:id=>id===generation};}
export function iou(a,b){
  const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection;
  return union>0?intersection/union:0;
}
export function decodeYolo(data,count,geometry,confidence=.1,overlap=.45){
  const {width,height,ratio,left,top}=geometry; const candidates=[];
  for(let i=0;i<count;i++){
    let score=0,classId=0;
    for(let c=0;c<80;c++){const s=data[(c+4)*count+i];if(s>score){score=s;classId=c;}}
    if(score<confidence)continue;
    const cx=data[i],cy=data[count+i],w=data[2*count+i],h=data[3*count+i];
    if(![cx,cy,w,h,score].every(Number.isFinite)||w<=0||h<=0)continue;
    candidates.push({class_id:classId,class_name:COCO_LABELS[classId],confidence:score,
      box_xyxy:[cx-w/2,cy-h/2,cx+w/2,cy+h/2]});
  }
  candidates.sort((a,b)=>b.confidence-a.confidence);const kept=[];
  for(const candidate of candidates){
    if(!kept.some(b=>b.class_id===candidate.class_id&&iou(b.box_xyxy,candidate.box_xyxy)>overlap))kept.push(candidate);
    if(kept.length>=300)break;
  }
  return kept.map(b=>({...b,box_xyxy:b.box_xyxy.map((v,i)=>Math.max(0,Math.min(i%2?height:width,(v-(i%2?top:left))/ratio)))}))
    .filter(b=>b.box_xyxy[2]>b.box_xyxy[0]&&b.box_xyxy[3]>b.box_xyxy[1]);
}
export const escapeXml=value=>String(value).replace(/[<>&"']/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&apos;'}[c]));
export function toCvatXml(records){
  const labels=[...new Set(records.flatMap(r=>r.boxes.map(b=>exportLabel(b,r.scope))))].sort();
  let xml='<?xml version="1.0" encoding="utf-8"?>\n<annotations><version>1.1</version><meta><task><name>M38 reviewed prelabels</name>';
  xml+=`<size>${records.length}</size><mode>annotation</mode><labels>`;
  xml+=labels.map(l=>`<label><name>${escapeXml(l)}</name><type>rectangle</type><attributes/></label>`).join('');
  xml+='</labels></task></meta>';
  records.forEach((r,index)=>{
    if(!(r.width>0&&r.height>0))throw new Error('Invalid image dimensions');
    xml+=`<image id="${index}" name="${escapeXml(r.name)}" width="${r.width}" height="${r.height}">`;
    r.boxes.forEach(b=>{
      const [x1,y1,x2,y2]=b.box_xyxy;
      if(![x1,y1,x2,y2].every(Number.isFinite)||!(0<=x1&&x1<x2&&x2<=r.width&&0<=y1&&y1<y2&&y2<=r.height))throw new Error('Invalid box');
      xml+=`<box label="${escapeXml(exportLabel(b,r.scope))}" occluded="0" source="${b.edited?'manual':'auto'}" xtl="${x1}" ytl="${y1}" xbr="${x2}" ybr="${y2}" z_order="0"/>`;
    });xml+='</image>';
  });return xml+'</annotations>';
}
