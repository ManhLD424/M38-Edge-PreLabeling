"""Canonical parameter fingerprint independent of torch checkpoint metadata."""
import hashlib
import json


def parameter_sha256(path):
    from ultralytics import YOLO
    model=YOLO(str(path)).model.float().cpu()
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
