from pathlib import Path
import pytest


def test_exporter_will_not_overwrite_existing_onnx(tmp_path):
    from src.core.compression.exporter import ModelExporter
    target=tmp_path/'onnx';target.mkdir()
    existing=target/'yolov8s.onnx';existing.write_bytes(b'preserve this artifact')
    exporter=object.__new__(ModelExporter)
    exporter.weights_path=Path('yolov8s.pt')
    exporter.output_root=tmp_path
    with pytest.raises(FileExistsError):
        exporter.export_onnx()
    assert existing.read_bytes()==b'preserve this artifact'
