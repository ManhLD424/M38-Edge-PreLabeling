"""Fetch the official checkpoint only if absent, verifying its recorded SHA-256."""
import hashlib
import json
from pathlib import Path
import tempfile
import requests
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    manifest=json.loads((ROOT/'docs/predictions/manifest.json').read_text(encoding='utf-8'))
    expected=manifest['models']['torch']['sha256']
    parameter_hash=manifest['models']['torch']['parameter_sha256']
    target=ROOT/'models/raw/yolov8s.pt'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            from src.core.inference.fingerprint import parameter_sha256
            if parameter_sha256(target)!=parameter_hash:
                raise ValueError('Existing checkpoint weights differ; it has been preserved')
        print('Checkpoint already present and verified')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt'
    with requests.get(url,stream=True,timeout=120) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.pt',delete=False) as output:
            temporary=Path(output.name)
            for chunk in response.iter_content(1024*1024):
                output.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
        from src.core.inference.fingerprint import parameter_sha256
        if parameter_sha256(temporary)!=parameter_hash:
            temporary.unlink()
            raise ValueError('Official download weights differ from replay checkpoint; refusing substitution')
        print('Checkpoint metadata/serialization differs; canonical model weights verified identical')
    temporary.replace(target)
    print('Downloaded and verified official YOLOv8s weights')


if __name__=='__main__':
    main()
