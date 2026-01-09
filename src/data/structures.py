
import torch
from dataclasses import dataclass, is_dataclass, fields
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from data.normalization import *
from utils.uv_tools import *

class TensorDict:
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __add__(self, other):
        cls = self.__class__
        added_fields = {}
        for f in fields(self):
            val1 = getattr(self, f.name)
            if isinstance(other, cls):
                val2 = getattr(other, f.name)
            else:
                val2 = other
            added_fields[f.name] = val1 + val2
        return cls(**added_fields)

    def __sub__(self, other):
        cls = self.__class__
        added_fields = {}
        for f in fields(self):
            val1 = getattr(self, f.name)
            if isinstance(other, cls):
                val2 = getattr(other, f.name)
            else:
                val2 = other
            added_fields[f.name] = val1 - val2
        return cls(**added_fields)

    def __mul__(self, other):
        cls = self.__class__
        added_fields = {}
        for f in fields(self):
            val1 = getattr(self, f.name)
            if isinstance(other, cls):
                val2 = getattr(other, f.name)
            else:
                val2 = other
            added_fields[f.name] = val1 * val2
        return cls(**added_fields)

    def __truediv__(self, other):
        cls = self.__class__
        added_fields = {}
        for f in fields(self):
            val1 = getattr(self, f.name)
            if isinstance(other, cls):
                val2 = getattr(other, f.name)
            else:
                val2 = other
            added_fields[f.name] = val1 / val2
        return cls(**added_fields)

    def keys(self):
        return (f.name for f in fields(self))

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def copy(self):
        cls = self.__class__
        copied_fields = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                copied_fields[f.name] = val.clone()
            elif hasattr(val, 'copy'):
                copied_fields[f.name] = val.copy()
            else:
                copied_fields[f.name] = val
        return cls(**copied_fields)
    
    def to(self, device=None, dtype=None, **kwargs):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Meshes):
                if device is not None:
                    setattr(self, f.name, val.to(device=device))
            elif hasattr(val, 'to'):
                if isinstance(val, torch.Tensor) and not torch.is_floating_point(val):
                    val = val.to(device=device, **kwargs)
                else:
                    val = val.to(device=device, dtype=dtype, **kwargs)
                setattr(self, f.name, val)
        return self

    def item(self):
        standard_fields = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                standard_fields[f.name] = val.item()
            elif hasattr(val, 'item'):
                standard_fields[f.name] = val.item()
            else:
                standard_fields[f.name] = val
        return standard_fields

@dataclass
class GarmentDict(TensorDict):
    betas: torch.Tensor
    poses: torch.Tensor
    trans: torch.Tensor
    bending: torch.Tensor
    stretching: torch.Tensor
    density: torch.Tensor
    root_joint_position: torch.Tensor = None
    cloth_vertices: torch.Tensor = None
    vdm: torch.Tensor = None    

@dataclass
class LazyDict(TensorDict):
    cloth_path: str
    smpl_seq_path: str
    frame_idx: int
    bending: float
    stretching: float
    density: float

@dataclass
class FittingDict(TensorDict):
    target_mesh: Meshes
    sampled_vertices: torch.Tensor
    sampled_normals: torch.Tensor
    betas: torch.Tensor
    poses: torch.Tensor
    trans: torch.Tensor
    root_joint_position: torch.Tensor
    body_vertices: torch.Tensor
    body_normals: torch.Tensor
    index:int

class Mesh:
    '''
        represent garment template mesh
    '''
    def __init__(self, config:dict, dtype=torch.float32, device="cpu", subdivide=False):
        '''
        Args:
            config (Dict): Configuration from json file.
            dtype and device
        '''
        self.device = device
        self.dtype = dtype
        self.img_size = config["model"]["image_size"]
        self.subdivide = subdivide
        if self.subdivide:
            self.path = config["subdivided_template_path"]
        else:
            self.path = config["template_path"]

        v, f, p = load_obj(self.path, False, device=self.device)
        self.v = v
        self.f = f.verts_idx
        self.vt = p.verts_uvs
        self.fvt = f.textures_idx

        self.vt2v, self.vt_occurrence = uv_to_vertex_mapping(self.v, self.f, self.vt, self.fvt)

        if (config["normalization"]["uv"]):
            self.vt = normalize_uv(self.vt, isometric=False)

        self.uv_face, self.uv_barycentric, self.mask = mesh_uv_barycentric(self.vt, self.fvt, self.img_size)
        self.position_map = mesh_uv_position(self.v, self.f, self.uv_face, self.uv_barycentric, self.img_size)
        self.correspondance = pixel_vertices_correspondance(self.uv_barycentric, self.uv_face, self.f)

    def trimesh(self):
        import trimesh
        return trimesh.Trimesh(self.v.cpu(), self.f.cpu())

    def __getitem__(self, key):
        return getattr(self, key)
    
def dataclass_collate(batch):
    elem = batch[0]

    if is_dataclass(elem):
        return type(elem)(**{
            f.name: dataclass_collate([getattr(b, f.name) for b in batch])
            for f in fields(elem)
        })
    elif isinstance(elem, Meshes):
        return join_meshes_as_batch(batch)
    return torch.utils.data.default_collate(batch)