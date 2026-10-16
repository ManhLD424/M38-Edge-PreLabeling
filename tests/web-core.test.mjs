import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decodeYolo, getReplay, toCvatXml, createRequestGate, visibleBoxes} from '../docs/assets/core.mjs';

test('YOLO decoding retains different classes and suppresses same-class duplicates',()=>{
  const output=new Float32Array(84*3);
  for(let i=0;i<3;i++){output[i]=320;output[3+i]=320;output[6+i]=100;output[9+i]=100;}
  output[4*3]=.9;output[6*3+1]=.8;output[4*3+2]=.7;
  const boxes=decodeYolo(output,3,{width:640,height:640,ratio:1,left:0,top:0},.25,.45);
  assert.deepEqual(boxes.map(b=>b.class_id),[0,2]);
  assert.deepEqual(boxes[0].box_xyxy,[270,270,370,370]);
});
test('missing replay image never receives another image predictions',()=>{
  assert.throws(()=>getReplay({images:{'BDD01.jpg':{runtimes:{torch:{detections:[]}}}}},'BDD06.jpg','torch'),/prediction/i);
});
test('XML exports exactly visible reviewed boxes and escapes user labels',()=>{
  const boxes=visibleBoxes([{class_name:'car & van',confidence:.8,box_xyxy:[1,2,30,40]},
    {class_name:'car',confidence:.2,box_xyxy:[50,60,70,80]}],.5,'all');
  const xml=toCvatXml([{name:'road<1>.jpg',width:1280,height:720,boxes}]);
  assert.equal((xml.match(/<box /g)||[]).length,1);
  assert.match(xml,/car &amp; van/);
  assert.match(xml,/road&lt;1&gt;.jpg/);
});
test('old inference responses cannot become the active result',()=>{
  const gate=createRequestGate();const old=gate.next();const current=gate.next();
  assert.equal(gate.current(old),false);assert.equal(gate.current(current),true);
});
test('COCO export preserves person while traffic export maps pedestrian',()=>{
  const boxes=[{class_name:'person',confidence:.9,box_xyxy:[1,2,30,40]}];
  assert.match(toCvatXml([{name:'x.jpg',width:100,height:100,scope:'all',boxes}]),/label="person"/);
  assert.match(toCvatXml([{name:'x.jpg',width:100,height:100,scope:'traffic',boxes}]),/label="pedestrian"/);
});
