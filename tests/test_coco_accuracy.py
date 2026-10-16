import json
import cv2
import numpy as np
import pytest
from src.core.inference.base import Detection
from tests.test_accuracy_bench import DummyEngine
from src.core.benchmark.coco_accuracy import evaluate_coco


def test_coco_ap_uses_ground_truth_category_ids_and_returns_perfect_ap(tmp_path):
    cv2.imwrite(str(tmp_path/'image.jpg'),np.zeros((100,100,3),np.uint8))
    annotations={'images':[{'id':12,'file_name':'image.jpg','width':100,'height':100}],
        'categories':[{'id':42,'name':'car'}],
        'annotations':[{'id':1,'image_id':12,'category_id':42,'bbox':[10,10,30,40],'area':1200,'iscrowd':0}]}
    path=tmp_path/'gt.json';path.write_text(json.dumps(annotations))
    det=Detection(10,10,40,50,.9,2,'car')
    result=evaluate_coco(DummyEngine([det]),path,tmp_path)
    assert result['map_50_95']==pytest.approx(1.)
    assert result['predictions'][0]['category_id']==42


def test_coco_ap_counts_empty_predictions_as_failure(tmp_path):
    cv2.imwrite(str(tmp_path/'image.jpg'),np.zeros((100,100,3),np.uint8))
    annotations={'images':[{'id':1,'file_name':'image.jpg','width':100,'height':100}],
        'categories':[{'id':2,'name':'car'}],
        'annotations':[{'id':1,'image_id':1,'category_id':2,'bbox':[1,1,30,40],'area':1200,'iscrowd':0}]}
    path=tmp_path/'gt.json';path.write_text(json.dumps(annotations))
    result=evaluate_coco(DummyEngine([]),path,tmp_path)
    assert result['map_50']==0
