"""Validate deployable artifacts without loading ML frameworks or calling a network."""
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def validate():
    manifest=json.loads((ROOT/'docs/predictions/manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema']=='m38-predictions-v2'
    split=json.loads((ROOT/'data/split_manifest.json').read_text())
    calib={r['sha256'] for r in split['calibration']}
    holdout={r['sha256'] for r in split['holdout']}
    assert not calib & holdout, 'Calibration/evaluation overlap'
    assert len(manifest['images'])==26
    for name,record in manifest['images'].items():
        image=ROOT/'docs/images'/name
        assert image.is_file(), name
        assert hashlib.sha256(image.read_bytes()).hexdigest()==record['sha256'], name
        assert record['sha256'] in calib|holdout
        assert set(record['runtimes'])==set(manifest['models'])
        assert record['width']>0 and record['height']>0
        for result in record['runtimes'].values():
            assert math.isfinite(result['latency_ms']) and result['latency_ms']>0
            for d in result['detections']:
                x1,y1,x2,y2=d['box_xyxy']
                assert all(math.isfinite(v) for v in (x1,y1,x2,y2,d['confidence']))
                assert 0<=x1<x2<=record['width'] and 0<=y1<y2<=record['height']
                assert 0<=d['class_id']<80 and .099<=d['confidence']<=1
    model=ROOT/'docs/models/yolov8s.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['models']['onnx']['sha256']
    thirdparty=json.loads((ROOT/'docs/vendor/manifest.json').read_text())
    for name,digest in thirdparty['files'].items():
        assert hashlib.sha256((ROOT/'docs/vendor'/name).read_bytes()).hexdigest()==digest, name
    html=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert 'SAMPLE_DATA' not in html and 'RUNTIME_METRICS' not in html
    assert 'assets/studio.mjs' in html and 'CVAT XML' in html
    print(f'Validated {len(manifest["images"])} images x {len(manifest["models"])} runtimes, split/model/vendor hashes and box bounds')


if __name__=='__main__':
    validate()
