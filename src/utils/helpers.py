import torch
import numpy as np
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import os
import re

# use alphanum_key to sort by natural order
_convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [_convert(c) for c in re.split('([0-9]+)', key)]

def batchify_dict(data:dict) -> dict:
    '''
    Add one batch dimension to an unbatched dictionary containing torch.Tensor
    '''
    for k in data.keys():
        if isinstance(data[k], dict):
            batchify_dict(data[k])
        elif isinstance(data[k], torch.Tensor):
            data[k] = data[k][None, ...]
    return data

def dict_to(data:dict, device, detach=False) -> dict:

    for k in data.keys():
        if isinstance(data[k], dict):
            dict_to(data[k])
        elif isinstance(data[k], torch.Tensor):
            if detach:
                data[k] = data[k].detach()
            data[k] = data[k].to(device)
    return data

# interpolate linearly from first to second tensor with step
def interpolate(first:torch.Tensor, second:torch.Tensor, steps:int) -> torch.Tensor:
    normalized_frames = torch.linspace(0, 1, steps, device=first.device)[None,...].T
    data_seq = torch.lerp(first, second, normalized_frames)
    return data_seq

# read custom binary file containing vertices position
def read_positions(filename, dtype=np.float32):
    with open(filename, "rb") as f:
        vertices = np.fromfile(f, dtype=dtype).reshape(-1,3)
    return torch.tensor(vertices)

# write custom binary file containing vertices position
def write_array(array:torch.Tensor, filename, dtype=np.float32):
    with open(filename, "wb") as f:
        array.detach().cpu().numpy().flatten().astype(dtype).tofile(f)

# sequence: list[trimesh.Trimesh]
def export_sequence(sequence, path: str, ext=".ply"):
    os.makedirs(path, exist_ok=True)

    for i, mesh in enumerate(sequence):
        mesh.export(os.path.join(path, "mesh_%.04d"%i + ext), include_attributes=False)

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    gpu_size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
            if isinstance(obj, torch.Tensor):
                gpu_size += obj.element_size() * obj.nelement()
        objects = get_referents(*need_referents)
    return size, gpu_size