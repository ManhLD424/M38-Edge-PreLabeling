"""Actual COCO AP requires human ground-truth annotations, never model predictions."""
import contextlib
import io
from pathlib import Path
import cv2


def evaluate_coco(engine, annotations, image_root, image_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from src.core.inference.base import COCO_TO_BDD100K
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(annotations))
    ids = sorted(image_ids if image_ids is not None else truth.getImgIds())
    if not ids or not truth.dataset.get('annotations'):
        raise ValueError('Nonempty human ground-truth COCO annotations are required')
    categories = {c['name']:c['id'] for c in truth.dataset['categories']}
    results = []
    root = Path(image_root).resolve()
    for image_id in ids:
        info = truth.loadImgs(image_id)[0]
        path = (root/info['file_name']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Annotation image path escapes image root')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable evaluation image: {path}')
        if image.shape[1] != info['width'] or image.shape[0] != info['height']:
            raise ValueError(f'GT dimensions do not match image: {path.name}')
        for det in engine.predict(image):
            category = categories.get(det.class_name, categories.get(COCO_TO_BDD100K.get(det.class_name)))
            if category is None:
                continue
            results.append({'image_id':image_id,'category_id':category,
                'bbox':[det.x1,det.y1,det.x2-det.x1,det.y2-det.y1],'score':det.confidence})
    with contextlib.redirect_stdout(io.StringIO()) as log:
        if results:
            predictions = truth.loadRes(results)
        else:
            predictions = COCO()
            predictions.dataset = {'images':truth.dataset['images'],'categories':truth.dataset['categories'],'annotations':[]}
            predictions.createIndex()
        evaluator = COCOeval(truth,predictions,'bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {'metric_kind':'COCO_AP_against_ground_truth','images':len(ids),
            'map_50_95':float(evaluator.stats[0]),'map_50':float(evaluator.stats[1]),
            'map_75':float(evaluator.stats[2]),'raw_summary':log.getvalue(),
            'predictions':results}
