"""Verify both exporters with an independent Datumaro CVAT importer.

Run using the main Python with --parser-python pointing to an isolated Python
environment containing datumaro==1.13.11. No network or CVAT credentials used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from src.core.cvat.export import export_cvat_xml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parser-python',required=True,type=Path)
    args=parser.parse_args()
    output=ROOT/'outputs/repair-2026-10-16'
    output.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='cvat-export-',dir=output))
    records=[{'image':'road & 1.jpg','width':1280,'height':720,'shapes':[
        {'label':'car & van','points':[1,2,30,40],'confidence':.8},
        {'label':'car','points':[50,60,70,80],'confidence':.2}]},
        {'image':'empty.jpg','width':1280,'height':720,'shapes':[]}]
    node_script="""
      import {visibleBoxes,toCvatXml} from './docs/assets/core.mjs';
      let input=''; for await (const chunk of process.stdin) input+=chunk;
      const {records,threshold}=JSON.parse(input);
      const converted=records.map(r=>({name:r.image,width:r.width,height:r.height,scope:'all',
        boxes:visibleBoxes(r.shapes.map(s=>({class_name:s.label,class_id:2,
          box_xyxy:s.points,confidence:s.confidence})),threshold,'all')}));
      process.stdout.write(toCvatXml(converted));
    """
    jobs=[]
    for exporter in ('python','browser'):
        for threshold in (.5,.9):
            directory=work/f'{exporter}-{threshold}'
            (directory/'images').mkdir(parents=True)
            for record in records:
                Image.new('RGB',(record['width'],record['height'])).save(directory/'images'/record['image'])
            if exporter=='python':
                xml=export_cvat_xml(records,conf_thresh=threshold)
            else:
                xml=subprocess.check_output(['node','--input-type=module','-e',node_script],
                    cwd=ROOT,input=json.dumps({'records':records,'threshold':threshold}),text=True)
            (directory/'annotations.xml').write_text(xml,encoding='utf-8')
            jobs.append({'path':str(directory),'exporter':exporter,'threshold':threshold,
                         'expected_boxes':1 if threshold==.5 else 0})
    jobfile=work/'jobs.json'
    jobfile.write_text(json.dumps(jobs),encoding='utf-8')
    importer="""
import json,sys
from pathlib import Path
from datumaro import Dataset, AnnotationType
results=[]
for job in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')):
    dataset=Dataset.import_from(job['path'],'cvat')
    actual=[]
    labels=dataset.categories().get(AnnotationType.label)
    for item in dataset:
        assert tuple(item.media.size)==(720,1280)
        for annotation in item.annotations:
            actual.append({'label':labels.items[annotation.label].name,'bbox':annotation.get_bbox()})
    assert len(dataset)==2
    assert len(actual)==job['expected_boxes'],(job,actual)
    if actual:
        assert actual==[{'label':'car & van','bbox':[1.,2.,29.,38.]}],actual
    results.append({'exporter':job['exporter'],'threshold':job['threshold'],
                    'images':len(dataset),'annotations':actual})
Path(sys.argv[2]).write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
"""
    result=work/'result.json'
    subprocess.run([str(args.parser_python.resolve()),'-c',importer,str(jobfile),str(result)],check=True,cwd=ROOT)
    print(f'Independent CVAT format verification saved: {result}')


if __name__=='__main__':
    main()
