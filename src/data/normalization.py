import math
import torch
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_multiply, axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_invert
from utils.representation import aa2six

BENDING_MIN = 1e-8
BENDING_MAX = 1e-4
STRETCHING_MIN = 40
STRETCHING_MAX = 200
DENSITY_MIN = 0.01
DENSITY_MAX = 0.7

material_limits = {
    'bending': [BENDING_MIN, BENDING_MAX],
    'stretching': [STRETCHING_MIN, STRETCHING_MAX],
    'density': [DENSITY_MIN, DENSITY_MAX],
}

def normalize_uv(uv:torch.Tensor, isometric=True):

    uv[:,0] -= torch.min(uv[:,0])
    uv[:,1] -= torch.min(uv[:,1])
    if isometric:
        uv = uv / torch.max(uv)
    else:
        uv[:,0] /= torch.max(uv[:,0])
        uv[:,1] /= torch.max(uv[:,1])
    
    return uv

def normalize_cloth(data, trans=True, rotation=True):

    # SMPL: v = lbs(poses) + trans
    vertices = data.cloth_vertices
    global_trans = data.trans[..., -1, :]

    if rotation:
        aa_rotation = data.poses[..., -1, :3]
        offset = data.root_joint_position

        rotation_matrix = axis_angle_to_matrix(aa_rotation).transpose(-1, -2)

        vertices -= offset
        vertices = rotation_matrix.matmul(vertices.transpose(-1, -2)).transpose(-1, -2)
        vertices += offset

    if trans:
        vertices -= global_trans
    
    data.cloth_vertices = vertices
    return vertices

def unnormalize_cloth(data, trans=True, rotation=True):

    vertices = data.cloth_vertices
    global_trans = data.trans[..., -1:, :]

    if trans:
        vertices += global_trans

    if rotation:
        aa_rotation = data['poses'][..., -1, :3]
        offset = data.root_joint_position

        rotation_matrix = axis_angle_to_matrix(aa_rotation)

        vertices -= offset
        vertices = rotation_matrix.matmul(vertices.transpose(-1, -2)).transpose(-1, -2)
        vertices += offset

    data.cloth_vertices = vertices
    return vertices

def get_normalized_body(data, pose='aa'):
    # data['trans'] = data['trans'][None,...].expand(3, -1, -1)
    # data['poses'] = data['poses'][None,...].expand(3, -1, -1)
    # normalize trans from the last frame (alias current frame)
    trans = data['trans'] - data['trans'][..., -1, :][..., None, :]

    # normalize poses from the last frame (alias current frame) TODO REMOVE THIS NORMALIZATION
    poses = data['poses'].reshape(*trans.shape[:-1], -1, 3)         # N x s x p x d
    root_orient = axis_angle_to_quaternion(poses[..., 0, :])                # N x s x d
    last_frame_inv = quaternion_invert(root_orient[..., -1, :])[..., None, :]  # N x 1s x d
    root_orient = quaternion_multiply(root_orient, last_frame_inv)          # N x s x d

    root_orient = quaternion_to_axis_angle(root_orient)
    poses = torch.cat((root_orient[..., None, :], poses[..., 1:, :]), dim=-2)

    if pose == 'six':
        poses = aa2six(poses)

    poses = poses.reshape(*trans.shape[:-1], -1)

    return trans, poses

 # Normalize x to the range [-1, 1]
def normalize(x:torch.Tensor, x_min, x_max, log=False):
    return 2 * (x - x_min) / (x_max - x_min) - 1

 # Normalize x to the range [-1, 1] after apply log
def log_normalize(x:torch.Tensor, x_min, x_max):
    x = torch.log(x + 1e-10)
    x_max = math.log(x_max)
    x_min = math.log(x_min)

    return normalize(x, x_min, x_max)

# Normalize, flatten and concat condition parameters to adapt to the cross attention input
def flatten_embeddings(data, material=True, do_normalize=True):
    trans, six_poses = get_normalized_body(data, 'six')

    static_embedding = data.betas
    if material:
        bending = data.bending
        stretching = data.stretching
        density = data.density
        if do_normalize:
            bending = normalize(bending, BENDING_MIN, BENDING_MAX) #TODO log_normalize would be better
            stretching = normalize(stretching, STRETCHING_MIN, STRETCHING_MAX)
            density = normalize(density, DENSITY_MIN, DENSITY_MAX)
        static_embedding = torch.cat((static_embedding, bending, stretching, density),dim=-1)

    trans = trans[..., :-1, :].flatten(-2)
    removed_zeros_poses = torch.cat((six_poses[..., :10*6], six_poses[..., 12*6:20*6]), dim=-1).flatten(-2)
    dynamic_embeddings = torch.cat((trans, removed_zeros_poses), dim=-1)

    embeddings = torch.cat((static_embedding, dynamic_embeddings), dim=-1)[..., None, :]

    return embeddings

def denormalize(grad, x_min, x_max):
    return 0.5 * (grad + 1) * (x_max - x_min) + x_min

def denormalize_material_grads(bending, stretching, density):
    with torch.no_grad():
        bending.grad = denormalize(bending.grad, BENDING_MIN, BENDING_MAX)
        stretching.grad = denormalize(stretching.grad, STRETCHING_MIN, STRETCHING_MAX)
        density.grad = denormalize(density.grad, DENSITY_MIN, DENSITY_MAX)

def denormalize_material(bending, stretching, density):
    return denormalize(bending, BENDING_MIN, BENDING_MAX), \
        denormalize(stretching, STRETCHING_MIN, STRETCHING_MAX), \
        denormalize(density, DENSITY_MIN, DENSITY_MAX)