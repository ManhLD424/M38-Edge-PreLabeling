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


def test_quantizer_will_not_overwrite_existing_ir(tmp_path):
    from src.core.compression.quantizer import ModelQuantizer
    existing=tmp_path/'yolov8s_int8.xml';existing.write_text('preserve')
    quantizer=object.__new__(ModelQuantizer)
    quantizer.xml_path=Path('yolov8s_fp16.xml')
    with pytest.raises(FileExistsError):
        quantizer.quantize_int8(str(tmp_path))
    assert existing.read_text()=='preserve'
