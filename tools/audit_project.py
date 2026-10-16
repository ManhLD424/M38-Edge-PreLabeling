"""Read-only audit probes; writes evidence only under outputs/audit-2026-10-16."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import threading
from http.server import HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import requests

OUT = ROOT / 'outputs/audit-2026-10-16'
OUT.mkdir(parents=True, exist_ok=True)
evidence = {}
url = 'https://manhld424.github.io/M38-Edge-PreLabeling/'
r = requests.get(url, timeout=30)
evidence['deployment'] = {'url': url, 'status': r.status_code,
    'sha256': hashlib.sha256(r.content).hexdigest(),
    'matches_docs_index': r.content == (ROOT/'docs/index.html').read_bytes()}
evidence['deployment']['matches_docs_index_normalized_newlines'] = r.content.decode('utf-8').replace('\r\n','\n') == (ROOT/'docs/index.html').read_text(encoding='utf-8')
(OUT/'deployed.html').write_bytes(r.content)

from src.core.inference.torch_engine import TorchInferenceEngine
from src.core.inference.ov_engine import OpenVINOInferenceEngine
from src.core.benchmark.accuracy_bench import compute_iou
ref = TorchInferenceEngine(str(ROOT/'models/raw/yolov8s.pt'))
fp = OpenVINOInferenceEngine(str(ROOT/'models/openvino_fp16/yolov8s_fp16.xml'))
int8 = OpenVINOInferenceEngine(str(ROOT/'models/openvino_int8/yolov8s_int8.xml'))

def agreement(a, b):
    used = set()
    matches = 0
    for d in sorted(a, key=lambda x: -x.confidence):
        candidates = [(compute_iou(d.box_xyxy, e.box_xyxy), i)
            for i, e in enumerate(b) if i not in used and d.class_id == e.class_id]
        if candidates:
            iou, i = max(candidates)
            if iou >= .5:
                matches += 1
                used.add(i)
    return matches

totals = {key: {'boxes': 0, 'matched_ref_at_iou50': 0} for key in ['fp16_bgr','fp16_rgb','int8_bgr','int8_rgb']}
rows = []
ref_total = 0
for p in sorted((ROOT/'data/sample_images').glob('*.jpg')):
    image = cv2.imread(str(p))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    baseline = ref.predict(image)
    ref_total += len(baseline)
    row = {'image': p.name, 'dimensions': list(image.shape[:2]), 'reference_boxes': len(baseline)}
    for key, engine, img in [('fp16_bgr',fp,image),('fp16_rgb',fp,rgb),('int8_bgr',int8,image),('int8_rgb',int8,rgb)]:
        dets = engine.predict(img)
        matched = agreement(baseline, dets)
        row[key] = {'boxes':len(dets),'matched_ref_at_iou50':matched}
        totals[key]['boxes'] += len(dets)
        totals[key]['matched_ref_at_iou50'] += matched
        if p.name in ('BDD01.jpg', 'BDD06.jpg'):
            vis = image.copy()
            for d in dets:
                x1,y1,x2,y2=map(int,d.box_xyxy)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imwrite(str(OUT/f'{p.stem}_{key}.jpg'),vis)
    rows.append(row)
evidence['color_ablation'] = {'reference_total':ref_total,'totals':totals,'images':rows,
    'note':'Model agreement only, not accuracy or mAP. RGB input is an audit-only channel swap; source unchanged.'}

# Demonstrate class-agnostic suppression with synthetic model output.
fake = np.zeros((1,84,2),dtype=np.float32)
fake[0,:4,:] = np.array([[320,320],[320,320],[100,100],[100,100]])
fake[0,4,0] = .9
fake[0,6,1] = .8
probe = object.__new__(OpenVINOInferenceEngine)
probe.conf_thresh=.25; probe.iou_thresh=.45; probe.output_layer='out'
probe.compiled_model=lambda _: {'out':fake}
evidence['nms_overlap_person_car'] = [d.class_name for d in probe.predict(np.zeros((640,640,3),np.uint8))]

from src.core.pointcloud.pcd_io import read_pcd
from src.core.models.detector import PointPillarsDetector, DetectorConfig
points = read_pcd(ROOT/'data/pointclouds/demo.pcd').xyz
detector = PointPillarsDetector(DetectorConfig(score_thresh=.99))
front=detector.detect(points)
full=detector.detect(points,full_scene=True)
evidence['3d']={'learned_model_loaded':detector.model is not None,'threshold':.99,
    'boxes':len(front),'scores':sorted(set(c.score for c in front)),
    'full_scene_equals_front':[c.to_dict() for c in front]==[c.to_dict() for c in full]}

evidence['calibration_overlap'] = sum(
    (ROOT/'data/calibration'/p.name).exists() and
    hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((ROOT/'data/calibration'/p.name).read_bytes()).digest()
    for p in (ROOT/'data/sample_images').glob('*.jpg'))

# Local ephemeral server: request public repo README only, no credentials/private files.
from src.web.app import WebMVPHandler
server=HTTPServer(('127.0.0.1',0),WebMVPHandler)
thread=threading.Thread(target=server.serve_forever,daemon=True)
thread.start()
try:
    traversal=requests.get(f'http://127.0.0.1:{server.server_port}/api/image',params={'name':'../../README.md'},timeout=5)
    evidence['local_path_traversal']={'status':traversal.status_code,
        'returned_repo_readme':traversal.content==(ROOT/'README.md').read_bytes()}
finally:
    server.shutdown(); server.server_close()

evidence['readme_commands']={}
for args in [ ['compress','--calib','data/calibration','--max-samples','25'],
    ['benchmark-2d','--warmup','5','--out','unused.json'],
    ['eval-accuracy','--data','data/sample_images','--out','unused.json'],
    ['serve','--host','127.0.0.1'] ]:
    result=subprocess.run([sys.executable,'-m','src.cli.main',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    evidence['readme_commands'][' '.join(args)]={'exit_code':result.returncode,'stderr':result.stderr[-500:]}
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='color_ablation'},indent=2,ensure_ascii=False))
print(json.dumps({'color_ablation': {'reference_total':ref_total,'totals':totals}},indent=2))
