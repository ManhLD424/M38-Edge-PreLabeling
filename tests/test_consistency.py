import cv2
import numpy as np
from src.core.inference.base import Detection
from tests.test_accuracy_bench import DummyEngine


def test_consistency_does_not_call_f1_map_and_requires_iou50(tmp_path):
    from src.core.benchmark.accuracy_bench import AccuracyBenchmark
    path = tmp_path/'image.jpg'
    cv2.imwrite(str(path),np.zeros((100,100,3),np.uint8))
    a = Detection(0,0,10,10,.9,2,'car')
    b = Detection(9,9,19,19,.8,2,'car')
    result = AccuracyBenchmark().evaluate_dataset(DummyEngine([a]),DummyEngine([b]),[path])
    assert 'map_50' not in result
    assert result['retention_at_50_percent'] == 0
    assert result['f1_at_50_percent'] == 0


def test_consistency_fails_on_unreadable_evaluation_images(tmp_path):
    import pytest
    from src.core.benchmark.accuracy_bench import AccuracyBenchmark
    with pytest.raises(ValueError,match='image'):
        AccuracyBenchmark().evaluate_dataset(DummyEngine([]),DummyEngine([]),[tmp_path/'missing.jpg'])
