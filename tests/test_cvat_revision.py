import pytest
from src.core.cvat.client import CVAT3DClient
from src.core.geometry.cvat_adapter import CVAT3DAdapter
from src.core.geometry.cuboid import Cuboid3D


def test_adapter_does_not_invent_server_revision():
    cuboid=Cuboid3D(0,0,0,4,2,1.5)
    payload=CVAT3DAdapter({'vehicles':10}).to_shapes_payload([cuboid])
    assert 'version' not in payload


@pytest.mark.parametrize('version',[None,True,'0',-1])
def test_invalid_revision_cannot_reach_http_boundary(version,monkeypatch):
    client=CVAT3DClient(token=None)
    monkeypatch.setattr(client.session,'patch',lambda *a,**kw:pytest.fail('Invalid revision reached HTTP boundary'))
    with pytest.raises(ValueError,match='version'):
        client.save_annotations(1,{'version':version,'shapes':[]})


def test_unauthenticated_annotation_write_never_reaches_network(monkeypatch):
    client=CVAT3DClient(token=None)
    monkeypatch.setattr(client.session,'patch',lambda *a,**kw:pytest.fail('Unauthenticated write reached HTTP boundary'))
    with pytest.raises(RuntimeError,match='token'):
        client.save_annotations(1,{'version':7,'shapes':[]})
