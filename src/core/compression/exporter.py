"""Export in an isolated temporary directory and preserve existing user artifacts."""
from pathlib import Path
import shutil
import tempfile


class ModelExporter:
    def __init__(self, weights_path, output_root='models'):
        self.weights_path = Path(weights_path).resolve()
        self.output_root = Path(output_root)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)

    def export_onnx(self, output_dir=None, opset=17):
        target = Path(output_dir) if output_dir else self.output_root/'onnx'
        destination = target/f'{self.weights_path.stem}.onnx'
        if destination.exists():
            raise FileExistsError(destination)
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory(prefix='m38-export-') as directory:
            temporary = Path(directory)/self.weights_path.name
            shutil.copy2(self.weights_path,temporary)
            exported = YOLO(str(temporary)).export(format='onnx',opset=opset,dynamic=False,
                                                 simplify=True,imgsz=640,half=False)
            target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(exported,destination)
        return destination

    def export_openvino_fp16(self, output_dir=None):
        target = Path(output_dir) if output_dir else self.output_root/'openvino_fp16'
        xml = target/f'{self.weights_path.stem}_fp16.xml'
        binary = xml.with_suffix('.bin')
        if xml.exists() or binary.exists():
            raise FileExistsError(f'Preserving existing model pair: {xml}')
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory(prefix='m38-export-') as directory:
            temporary = Path(directory)/self.weights_path.name
            shutil.copy2(self.weights_path,temporary)
            exported = Path(YOLO(str(temporary)).export(format='openvino',half=True,imgsz=640))
            source_xml = next(exported.glob('*.xml'))
            target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_xml,xml)
            shutil.copy2(source_xml.with_suffix('.bin'),binary)
        return xml
