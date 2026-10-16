"""Point Cloud Data (PCD) file parser and writer."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PCDData:
    """Parsed Point Cloud Data representation."""
    points: np.ndarray  # Shape (N, 3) or (N, 4)
    fields: List[str]
    count: int
    header: Dict[str, Any]
    intensity: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None

    @property
    def xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of shape (N, 3)."""
        return self.points[:, :3]

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Returns (xmin, ymin, zmin, xmax, ymax, zmax)."""
        if self.count == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        xyz = self.xyz
        return (
            float(xyz[:, 0].min()),
            float(xyz[:, 1].min()),
            float(xyz[:, 2].min()),
            float(xyz[:, 0].max()),
            float(xyz[:, 1].max()),
            float(xyz[:, 2].max()),
        )


def read_pcd(path: Union[str, Path]) -> PCDData:
    """Parse binary/ASCII PCD with SIZE/TYPE/COUNT and named XYZ fields."""
    import io
    stream = io.BytesIO(Path(path).read_bytes())
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('PCD missing DATA header')
        parts = line.decode('ascii').strip().split()
        if not parts or parts[0].startswith('#'):
            continue
        header[parts[0].upper()] = parts[1:]
        if parts[0].upper() == 'DATA':
            break
    fields = header.get('FIELDS', [])
    if not all(axis in fields for axis in ('x','y','z')) or len(set(fields)) != len(fields):
        raise ValueError('PCD requires unique x/y/z fields')
    sizes = list(map(int,header.get('SIZE',['4']*len(fields))))
    types = header.get('TYPE',['F']*len(fields))
    counts = list(map(int,header.get('COUNT',['1']*len(fields))))
    if not len(fields)==len(sizes)==len(types)==len(counts) or any(c<=0 for c in counts):
        raise ValueError('Invalid PCD field schema')
    count = int(header.get('POINTS',[int(header.get('WIDTH',[0])[0])*int(header.get('HEIGHT',[1])[0])])[0])
    if count < 0:
        raise ValueError('Invalid PCD point count')
    mapping = {('F',4):'<f4',('F',8):'<f8',('I',1):'i1',('I',2):'<i2',('I',4):'<i4',
               ('U',1):'u1',('U',2):'<u2',('U',4):'<u4'}
    try:
        dtype = np.dtype([(name,mapping[(kind,size)],(n,)) if n>1 else (name,mapping[(kind,size)])
                          for name,size,kind,n in zip(fields,sizes,types,counts)])
    except KeyError as exc:
        raise ValueError('Unsupported PCD numeric type') from exc
    body = stream.read()
    mode = header['DATA'][0]
    if mode == 'binary':
        if len(body) < count*dtype.itemsize:
            raise ValueError('PCD binary data truncated')
        data = np.frombuffer(body[:count*dtype.itemsize],dtype=dtype,count=count)
        columns = {name:data[name] for name in fields}
    elif mode == 'ascii':
        if count == 0:
            columns = {name:np.empty((0,)) for name in fields}
        else:
            values = np.loadtxt(io.BytesIO(body),dtype=np.float64,ndmin=2)
            if values.shape != (count,sum(counts)):
                raise ValueError('PCD ASCII dimensions do not match header')
            columns = {};offset=0
            for name,n in zip(fields,counts):
                columns[name] = values[:,offset] if n==1 else values[:,offset:offset+n]
                offset += n
    else:
        raise ValueError(f'Unsupported PCD DATA encoding: {mode}')
    if any(counts[fields.index(axis)]!=1 for axis in ('x','y','z')):
        raise ValueError('XYZ fields must be scalars')
    xyz = np.column_stack([columns[axis] for axis in ('x','y','z')]).astype(np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError('PCD contains non-finite XYZ values')
    return PCDData(points=xyz,fields=fields,count=count,header=header,
                   intensity=columns.get('intensity'),rgb=columns.get('rgb'))


def write_pcd(path: Union[str, Path], points: np.ndarray, binary: bool = True) -> None:
    """Writes point coordinates to a PCD file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "# .PCD v0.7 - Exported by M38-Edge-PreLabeling\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )
    if binary:
        xyz_bytes = points[:, :3].astype(np.float32).tobytes()
        with p.open("wb") as f:
            f.write(header.encode("ascii"))
            f.write(xyz_bytes)
    else:
        with p.open("w", encoding="utf-8") as f:
            f.write(header)
            for pt in points[:, :3]:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
