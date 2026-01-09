import trimesh
import torch
import json
from body_models.smpl_model import SMPLH, SMPLH_JOINTS_BODY
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.io import load_objs_as_meshes, save_obj
with open('../configs/config.json', 'rb') as f:
    config = json.load(f)
body_model = SMPLH(config)
body_model.eval().to('cpu')
data = {
    'betas': torch.zeros((1, body_model.num_betas)),
    'trans': torch.zeros((1, 3)),
    'poses': torch.zeros((1, SMPLH_JOINTS_BODY*3))
}
bv, j = body_model(data, Jtr=True)
offset = j[0,0]

src = '/home/adumouli/Data/MY_DATASET/Dataset/'
dst = '/home/adumouli/Dev/EXPERIMENT/hood_data/aux_data/garment_meshes/'
name = 'Cos5kZero_subdivided'
# name = 'Cos5kZero'
# name = 'Cos15kZero'
# mesh = trimesh.load_mesh(root + 'Cos5kZero_subdivided.obj', process=False)
# verts = torch.tensor(mesh.vertices).float()

mesh = load_objs_as_meshes([src + name + '.obj'], load_textures=False)
verts = mesh.verts_packed()
rotation_matrix = axis_angle_to_matrix(torch.tensor([-torch.pi*.5, 0.0, 0.0]))  # Rotate -90 degrees around x-axis
verts -= offset
verts = rotation_matrix.matmul(verts.transpose(-1, -2)).transpose(-1, -2)
verts += offset

save_obj(dst + name + '.obj', verts, mesh.faces_packed())

# mesh.export(root + 'Cos5kZero_subdivided_fixed.obj')
