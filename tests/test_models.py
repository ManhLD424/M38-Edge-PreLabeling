"""Unit tests for deliverable ONNX model integrity and graph validity."""
import unittest
from pathlib import Path
import onnx
import onnxruntime as ort

class TestDeliverableModels(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.model_files = [
            "yolo26n_fp16.onnx",
            "yolo26s_fp16.onnx",
            "yolo26m_fp16.onnx"
        ]

    def test_models_exist(self):
        """Verify all three deliverable ONNX models are present."""
        for mf in self.model_files:
            p = self.repo_root / mf
            self.assertTrue(p.exists(), f"Model file missing: {mf}")
            self.assertGreater(p.stat().st_size, 1024 * 1024, f"Model file too small: {mf}")

    def test_onnx_checker(self):
        """Verify models pass ONNX schema and graph validation."""
        for mf in self.model_files:
            p = self.repo_root / mf
            proto = onnx.load(str(p))
            onnx.checker.check_model(proto)

    def test_onnx_runtime_session(self):
        """Verify models can be loaded into an ONNX Runtime InferenceSession."""
        for mf in self.model_files:
            p = self.repo_root / mf
            session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            self.assertGreaterEqual(len(inputs), 1)
            self.assertGreaterEqual(len(outputs), 1)
            self.assertEqual(inputs[0].name, "images")

if __name__ == "__main__":
    unittest.main()
