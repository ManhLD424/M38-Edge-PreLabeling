"""Unit tests for YOLO and COCO label exporters."""
import unittest
import json
from pathlib import Path
import tempfile
from src.prelabel.engine import Detection, InferenceResult
from src.prelabel.exporter import export_to_yolo, export_to_coco

class TestExporters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        # Mock detections
        d1 = Detection(
            class_id=2,
            class_name="car",
            confidence=0.88,
            bbox_xyxy=[100.0, 150.0, 300.0, 350.0],
            bbox_xywhn=[0.25, 0.35, 0.20, 0.25],
            status="AUTO_APPROVED"
        )
        d2 = Detection(
            class_id=0,
            class_name="person",
            confidence=0.55,
            bbox_xyxy=[50.0, 60.0, 90.0, 180.0],
            bbox_xywhn=[0.07, 0.12, 0.04, 0.12],
            status="REVIEW_NEEDED"
        )

        self.mock_result = InferenceResult(
            image_path="sample.jpg",
            image_shape=(720, 1280),
            detections=[d1, d2],
            latency_ms=25.0,
            fps=40.0,
            auto_approved_count=1,
            review_needed_count=1
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_to_yolo(self):
        """Verify YOLO .txt file generation and format."""
        out_txt = self.out_dir / "sample.txt"
        export_to_yolo(self.mock_result, str(out_txt), include_confidence=True)
        
        self.assertTrue(out_txt.exists())
        lines = out_txt.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)
        
        tokens = lines[0].split()
        self.assertEqual(tokens[0], "2") # car class_id
        self.assertAlmostEqual(float(tokens[1]), 0.25, places=2)
        self.assertAlmostEqual(float(tokens[5]), 0.88, places=2)

    def test_export_to_coco(self):
        """Verify COCO JSON format generation and schema integrity."""
        out_json = self.out_dir / "annotations.json"
        export_to_coco([self.mock_result], str(out_json))
        
        self.assertTrue(out_json.exists())
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("images", data)
        self.assertIn("annotations", data)
        self.assertIn("categories", data)
        self.assertEqual(len(data["images"]), 1)
        self.assertEqual(len(data["annotations"]), 2)
        self.assertEqual(data["annotations"][0]["category_id"], 2)
        self.assertEqual(data["annotations"][0]["status"], "AUTO_APPROVED")

if __name__ == "__main__":
    unittest.main()
