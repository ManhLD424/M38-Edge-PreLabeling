import struct
import numpy as np
import pytest
from src.core.pointcloud.pcd_io import read_pcd


def test_reads_crlf_reordered_float64_coordinates_and_uint16_intensity(tmp_path):
    path=tmp_path/'scan.pcd'
    header='FIELDS y x z intensity\r\nSIZE 8 8 8 2\r\nTYPE F F F U\r\nCOUNT 1 1 1 1\r\nWIDTH 1\r\nHEIGHT 1\r\nPOINTS 1\r\nDATA binary\r\n'
    path.write_bytes(header.encode()+struct.pack('<dddH',2.,1.,3.,65530))
    cloud=read_pcd(path)
    np.testing.assert_allclose(cloud.xyz,[[1,2,3]])
    assert cloud.intensity[0]==65530


def test_truncated_binary_cloud_is_rejected_not_silently_shortened(tmp_path):
    path=tmp_path/'truncated.pcd'
    path.write_bytes(b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 2\nDATA binary\n'+struct.pack('<fff',1,2,3))
    with pytest.raises(ValueError,match='truncated'):
        read_pcd(path)
