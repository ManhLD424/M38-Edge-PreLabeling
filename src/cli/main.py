"""M38 commands: real inference, CVAT XML export, consistency and GT-based COCO AP."""
import hashlib
import json
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import click
import cv2
from src.core.inference.factory import make_engine, MODELS, ROOT


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def split_paths(manifest, group, image_root):
    data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    hashes = {key:{item['sha256'] for item in data[key]} for key in ('calibration','holdout')}
    if hashes['calibration'] & hashes['holdout']:
        raise click.ClickException('Calibration/holdout overlap by content hash')
    root = Path(image_root).resolve()
    paths = []
    for item in data[group]:
        path = (root/item['name']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise click.ClickException('Missing/invalid split image')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise click.ClickException(f'Split hash mismatch: {path.name}')
        paths.append(path)
    if not paths:
        raise click.ClickException('Empty split')
    return paths


@click.group()
def cli():
    """M38 Edge Prelabel Studio. Run from repository root."""


@cli.command()
def info():
    """Report actual host, Python and installed model artifacts."""
    click.echo(json.dumps({'python':platform.python_version(),'os':platform.platform(),
        'models':{key:{'path':path,'available':Path(path).is_file()} for key,path in MODELS.items()}},indent=2))


@cli.command()
@click.option('--weights',default='models/raw/yolov8s.pt',type=click.Path(exists=True,dir_okay=False))
@click.option('--split-manifest',default='data/split_manifest.json',type=click.Path(exists=True))
@click.option('--image-root',default='data/sample_images',type=click.Path(exists=True))
@click.option('--output-root',default='models/rebuilt')
def compress(weights,split_manifest,image_root,output_root):
    """Rebuild ONNX/FP16/INT8 using frozen calibration images, never random noise."""
    from src.core.compression.exporter import ModelExporter
    from src.core.compression.quantizer import ModelQuantizer
    paths = split_paths(split_manifest,'calibration',image_root)
    if Path(output_root).exists():
        raise click.ClickException('Output root already exists; choose a new directory to preserve artifacts')
    exporter = ModelExporter(weights,output_root)
    onnx = exporter.export_onnx()
    fp16 = exporter.export_openvino_fp16()
    with tempfile.TemporaryDirectory(prefix='m38-calibration-') as directory:
        for path in paths:
            shutil.copy2(path,Path(directory)/path.name)
        int8 = ModelQuantizer(str(fp16)).quantize_int8(str(Path(output_root)/'openvino_int8_rgb'),directory,len(paths))
    write_json(Path(output_root)/'provenance.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        'split_manifest_sha256':hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest(),
        'preprocessing':'RGB letterbox 640, padding 114, float32 NCHW /255',
        'onnx':str(onnx),'fp16':str(fp16),'int8':str(int8)})
    click.echo(f'Rebuilt artifacts: {output_root}; update model configuration explicitly before serving.')


@cli.command(name='prelabel-2d')
@click.option('--source',default='data/sample_images',type=click.Path(exists=True))
@click.option('--model',default=MODELS['openvino_fp16'],type=click.Path(exists=True,dir_okay=False))
@click.option('--out',default='outputs/prelabels-v2')
@click.option('--conf',default=.25,type=click.FloatRange(0,1))
@click.option('--scope',type=click.Choice(['traffic','all']),default='traffic')
def prelabel_2d(source,model,out,conf,scope):
    """Infer real boxes and export CVAT for images 1.1 XML plus an explicit custom JSON."""
    from src.core.cvat.export import export_cvat_xml
    from src.core.inference.base import COCO_TO_BDD100K
    runtime = 'openvino_fp16' if model.endswith('.xml') else 'onnx' if model.endswith('.onnx') else 'torch'
    engine = make_engine(runtime,model,conf)
    path = Path(source)
    images = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not images:
        raise click.ClickException('No images found')
    records = []
    target = Path(out)
    target.mkdir(parents=True,exist_ok=True)
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise click.ClickException(f'Unreadable image: {image_path}')
        start = time.perf_counter()
        detections = engine.predict(image)
        shapes = []
        for det in detections:
            if scope=='traffic' and det.class_name not in COCO_TO_BDD100K:
                continue
            label = COCO_TO_BDD100K.get(det.class_name,det.class_name) if scope=='traffic' else det.class_name
            shapes.append({'type':'rectangle','label':label,'points':det.box_cvat,'confidence':det.confidence})
            x1,y1,x2,y2 = map(int,det.box_xyxy)
            cv2.rectangle(image,(x1,y1),(x2,y2),(40,220,140),2)
            cv2.putText(image,label,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,220,140),1)
        records.append({'image':image_path.name,'width':image.shape[1],'height':image.shape[0],
            'shapes':shapes,'elapsed_with_render_ms':round((time.perf_counter()-start)*1000,2)})
        cv2.imwrite(str(target/f'pred_{image_path.name}'),image)
    (target/'annotations.xml').write_text(export_cvat_xml(records),encoding='utf-8')
    write_json(target/'studio_prelabels.json',{'schema':'m38-prelabels-v1','images':records})
    click.echo(f'Exported {len(records)} images: {target}/annotations.xml (CVAT for images 1.1)')


@cli.command(name='compare-runtimes')
@click.option('--split-manifest',default='data/split_manifest.json',type=click.Path(exists=True))
@click.option('--image-root',default='data/sample_images',type=click.Path(exists=True))
@click.option('--save-json',default='outputs/consistency-v2.json')
def compare_runtimes(split_manifest,image_root,save_json):
    """Measure model agreement on a held-out image split. This is NOT accuracy/mAP."""
    from src.core.benchmark.accuracy_bench import AccuracyBenchmark
    paths = split_paths(split_manifest,'holdout',image_root)
    reference = make_engine('torch')
    results = {}
    for runtime in ('onnx','openvino_fp16','openvino_int8'):
        results[runtime] = AccuracyBenchmark().evaluate_dataset(reference,make_engine(runtime),paths)
    write_json(save_json,{'split':split_manifest,'evaluations':results})
    click.echo(json.dumps(results,indent=2))


@cli.command(name='eval-accuracy')
@click.option('--annotations',required=True,type=click.Path(exists=True,dir_okay=False),help='Human ground truth, COCO instances JSON')
@click.option('--image-root',required=True,type=click.Path(exists=True,file_okay=False))
@click.option('--runtime',type=click.Choice(list(MODELS)),default='openvino_fp16')
@click.option('--save-json',default='outputs/coco-accuracy.json')
def eval_accuracy(annotations,image_root,runtime,save_json):
    """Evaluate actual COCO AP against human GT. Predictions are not ground truth."""
    from src.core.benchmark.coco_accuracy import evaluate_coco
    engine = make_engine(runtime,conf=.001)
    metrics = evaluate_coco(engine,annotations,image_root)
    metrics['annotation_sha256'] = hashlib.sha256(Path(annotations).read_bytes()).hexdigest()
    write_json(save_json,metrics)
    click.echo(json.dumps({key:value for key,value in metrics.items() if key!='predictions'},indent=2))


@cli.command(name='benchmark-2d')
@click.option('--runs',default=3,type=click.IntRange(1,100))
@click.option('--warmup',default=3,type=click.IntRange(0,100))
@click.option('--sample-dir',default='data/sample_images',type=click.Path(exists=True))
@click.option('--save-json','--out',default='outputs/latency-v2.json')
def benchmark_2d(runs,warmup,sample_dir,save_json):
    """Time sequential end-to-end predict() on real images; retain raw samples."""
    import psutil
    paths = sorted(Path(sample_dir).glob('*.jpg'))
    if not paths:
        raise click.ClickException('No benchmark images')
    images = [cv2.imread(str(p)) for p in paths]
    if any(image is None for image in images):
        raise click.ClickException('Unreadable benchmark image')
    import numpy as np
    results = {}
    for runtime in MODELS:
        engine = make_engine(runtime)
        for _ in range(warmup):
            engine.predict(images[0])
        raw = []
        for repeat in range(runs):
            for path,image in zip(paths,images):
                start = time.perf_counter_ns(); detections = engine.predict(image)
                raw.append({'image':path.name,'repeat':repeat,'boxes':len(detections),
                    'ms':(time.perf_counter_ns()-start)/1e6})
        times = np.array([r['ms'] for r in raw])
        results[runtime] = {'mean_ms':float(times.mean()),'p50_ms':float(np.percentile(times,50)),
            'p95_ms':float(np.percentile(times,95)),'p99_ms':float(np.percentile(times,99)),
            'sequential_images_per_second':1000/float(times.mean()),'raw':raw,
            'model_path':str(engine.model_path),'model_sha256':hashlib.sha256(Path(engine.model_path).read_bytes()).hexdigest()}
    from src.core.inference.settings import CPU_THREADS
    import importlib.metadata as metadata
    write_json(save_json,{'created_at':datetime.now(timezone.utc).isoformat(),
        'hardware':platform.platform(),'cpu':platform.processor(),'hardware_threads':psutil.cpu_count(),
        'inference_threads':CPU_THREADS,'runs_per_image':runs,'warmup_runs':warmup,
        'versions':{p:metadata.version(p) for p in ('torch','ultralytics','onnxruntime','openvino','numpy','opencv-python')},
        'protocol':'sequential predict including preprocess/NMS; no streaming throughput claim',
        'images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'evaluations':results})
    click.echo(f'Wrote raw real-image timings: {save_json}')


@cli.command()
@click.option('--port',default=8080,type=click.IntRange(1,65535))
@click.option('--host',default='127.0.0.1',type=click.Choice(['127.0.0.1','::1']))
def serve(port,host):
    """Serve the shared Studio and CPU inference API on loopback only."""
    from src.web.app import run_server
    run_server(port,host)


@cli.command(name='infer-3d')
@click.option('--pcd',default='data/pointclouds/demo.pcd',type=click.Path(exists=True))
@click.option('--out',default='outputs/3d-heuristic')
@click.option('--score-thresh',default=.3,type=click.FloatRange(0,1))
@click.option('--full-scene',is_flag=True)
@click.option('--checkpoint',type=click.Path(exists=True,dir_okay=False))
@click.option('--config',default='configs/default_config.yaml',type=click.Path(exists=True,dir_okay=False))
def infer_3d(pcd,out,score_thresh,full_scene,checkpoint,config):
    """Experimental geometry proposals, or explicit learned checkpoint; no CVAT IDs guessed."""
    from src.core.models.detector import PointPillarsDetector, DetectorConfig
    from src.core.pointcloud.pcd_io import read_pcd
    points = read_pcd(pcd)
    cloud = points.xyz
    if points.intensity is not None:
        import numpy as np
        cloud = np.column_stack((cloud,points.intensity))
    import yaml
    settings = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    model = settings['pointpillars']
    classes = [name for _,name in sorted(settings['classes'].items(),key=lambda pair:int(pair[0]))]
    detector = PointPillarsDetector(DetectorConfig(score_thresh=score_thresh,
        checkpoint_path=checkpoint or model.get('checkpoint'),classes=classes,
        point_cloud_range=tuple(settings['pointcloud']['range']),voxel_size=tuple(model['voxel_size']),
        delta=float(model['delta']),heading_mode=model['heading_mode'],z_bottom=bool(model['z_bottom'])))
    boxes = detector.detect(cloud,full_scene=full_scene)
    backend = detector.backend
    write_json(Path(out)/'proposals.json',{'schema':'m38-3d-proposals-v1','backend':backend,
        'quality':'unvalidated; scores of heuristic are not calibrated probabilities',
        'boxes':[box.to_dict() for box in boxes]})
    click.echo(f'{backend}: {len(boxes)} proposals; custom JSON, not a CVAT import file')


@cli.command(name='benchmark-3d')
@click.option('--pcd',default='data/pointclouds/demo.pcd',type=click.Path(exists=True))
@click.option('--runs',default=3,type=click.IntRange(1,100))
def benchmark_3d_cmd(pcd,runs):
    """Time experimental geometry proposals, not learned PointPillars inference."""
    from dataclasses import asdict
    from src.core.benchmark.benchmark_3d import Benchmark3D
    result = asdict(Benchmark3D().run(pcd,runs=runs))
    result['backend'] = 'experimental_geometry_heuristic'
    result['rss_note'] = 'RSS before/after only; not peak memory'
    click.echo(json.dumps(result,indent=2))


if __name__=='__main__':
    cli()
