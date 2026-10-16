import cv2
import numpy as np
import pytest
import xml.etree.ElementTree as ET
from click.testing import CliRunner
from src.cli.main import cli
from src.core.inference.base import Detection
from tests.test_accuracy_bench import DummyEngine


@pytest.mark.parametrize('scope,label',[('all','person'),('traffic','pedestrian')])
def test_cli_exports_selected_ontology(tmp_path,monkeypatch,scope,label):
    import src.cli.main as main
    image=tmp_path/'image.jpg';cv2.imwrite(str(image),np.zeros((100,100,3),np.uint8))
    model=tmp_path/'dummy.pt';model.write_bytes(b'dummy external inference boundary')
    monkeypatch.setattr(main,'make_engine',lambda *args,**kwargs:DummyEngine([Detection(1,2,30,40,.9,0,'person')]))
    out=tmp_path/'export'
    result=CliRunner().invoke(cli,['prelabel-2d','--source',str(image),'--model',str(model),'--out',str(out),'--scope',scope])
    assert result.exit_code==0,result.output
    assert ET.parse(out/'annotations.xml').getroot().find('image/box').get('label')==label
