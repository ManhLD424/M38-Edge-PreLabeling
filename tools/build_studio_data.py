"""Generate image-specific replay data from real local engines with provenance."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
from src.core.inference.factory import MODELS, make_engine


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    manifest={'schema':'m38-predictions-v2','generated_at':datetime.now(timezone.utc).isoformat(),
        'hardware':{'platform':platform.platform(),'processor':platform.processor()},
        'preprocessing':'RGB letterbox 640x640, padding114, class-aware NMS IoU .45, confidence floor .10, max_det300',
        'quality':'Model predictions, NOT ground truth. No validated AP or annotator time saving.',
        'models':{},'images':{}}
    images=sorted((ROOT/'data/sample_images').glob('*.jpg'))
    split=json.loads((ROOT/'data/split_manifest.json').read_text())
    calibration={p['sha256'] for p in split['calibration']}
    for path in images:
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        image_hash=digest(path)
        manifest['images'][path.name]={'sha256':image_hash,'width':image.shape[1],'height':image.shape[0],
            'split':'calibration' if image_hash in calibration else 'holdout','runtimes':{}}
    for runtime,relative in MODELS.items():
        model=ROOT/relative
        if not model.is_file():
            raise FileNotFoundError(f'Missing {runtime}: {model}')
        files=[model]+([model.with_suffix('.bin')] if model.suffix=='.xml' else [])
        manifest['models'][runtime]={'path':relative,'sha256':digest(model),
            'files':{p.name:digest(p) for p in files},'size_bytes':sum(p.stat().st_size for p in files),
            'quality_note':'Illustrative split only, no independent scene metadata or GT. INT8 was recalibrated on the calibration split.' if runtime=='openvino_int8' else 'COCO pretrained model; not BDD-trained or validated against GT here.'}
        engine=make_engine(runtime,conf=.1)
        engine.predict(cv2.imread(str(images[0])))  # explicit warmup before timings
        for path in images:
            image=cv2.imread(str(path))
            start=time.perf_counter()
            detections=engine.predict(image)
            elapsed=(time.perf_counter()-start)*1000
            manifest['images'][path.name]['runtimes'][runtime]={'latency_ms':round(elapsed,3),
                'detections':[{'class_id':d.class_id,'class_name':d.class_name,'confidence':d.confidence,
                               'box_xyxy':list(d.box_xyxy)} for d in detections]}
        print(f'{runtime}: {len(images)} image-specific predictions',flush=True)
    out=ROOT/'docs/predictions'
    out.mkdir(exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Wrote docs/predictions/manifest.json',flush=True)


if __name__=='__main__':
    build()
