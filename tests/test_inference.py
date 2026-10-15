"""Unit tests for EdgePreLabelEngine inference and confidence filtering."""
import unittest
from pathlib import Path
import numpy as np
from src.prelabel.engine import EdgePreLabelEngine, Detection, InferenceResult

class TestEdgeInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.nano_model = cls.repo_root / "yolo26n_fp16.onnx"
        if not cls.nano_model.exists():
            raise unittest.SkipTest(f"Nano model not found at: {cls.nano_model}")
        cls.engine = EdgePreLabelEngine(str(cls.nano_model), device="cpu")

    def test_predict_dummy_image(self):
        """Test inference pipeline execution on a synthetic image."""
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw dummy shape
        dummy_img[100:300, 200:500] = 200
        
        result = self.engine.predict(dummy_img, image_path="test_dummy.jpg")
        
        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.image_shape, (720, 1280))
        self.assertGreater(result.latency_ms, 0.0)
        self.assertGreater(result.fps, 0.0)
        self.assertIsInstance(result.detections, list)

    def test_letterbox_dimensions(self):
        """Test letterbox preprocessing output shapes."""
        img = np.zeros((480, 854, 3), dtype=np.uint8)
        tensor, ratio, (dw, dh) = self.engine.preprocess(img)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertTrue(np.issubdtype(tensor.dtype, np.floating))

if __name__ == "__main__":
    unittest.main()
