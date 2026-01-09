import torch
import numpy as np
from pytorch3d.ops.knn import knn_points, knn_gather
import math

# from https://github.com/pytorch/pytorch/issues/59194
def angle_v2(a, b, dim=-1):
    a_norm = torch.norm(a, p=2, dim=dim, keepdim=True)
    b_norm = torch.norm(b, p=2, dim=dim, keepdim=True)
    return 2 * torch.atan2(
        torch.norm((a * b_norm - a_norm * b), p=2, dim=dim),
        torch.norm((a * b_norm + a_norm * b), p=2, dim=dim)
    )*180/math.pi

# from https://github.com/facebookresearch/pytorch3d/issues/736
def compute_points_normals(points, faces):
    verts_normals = torch.zeros_like(points)
    vertices_faces = points[faces]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces[:, 0], faces_normals)
    verts_normals.index_add_(0, faces[:, 1], faces_normals)
    verts_normals.index_add_(0, faces[:, 2], faces_normals)
    
    return verts_normals

'''
Ecloth from {Estimation of Human Body Shape in Motion with Wide Clothing, J. Yang}
'''
def clothing_energy(body_vertices, cloth_vertices, cloth_normals, body_normals):

    # ∑Nv δout δNN knn.dists
    knn = knn_points(body_vertices, cloth_vertices)
    dists = knn.dists

    # gather knn values
    cloth_knn_normals = knn_gather(cloth_normals, knn.idx)[:,:,0]
    cloth_knn_vertices = knn_gather(cloth_vertices, knn.idx)[:,:,0]

    # compute δNN (outliers), siplified version, should compute smpl normals
    angle = angle_v2(cloth_knn_normals, body_normals)

    dists[angle>=60] = 0
    dists[dists>=0.2] = 0

    # compute δout
    cloth_smpl_vector = body_vertices - cloth_knn_vertices
    angle = angle_v2(cloth_knn_normals, cloth_smpl_vector)
    dists[angle>=90] = 0    # inside

    loss = torch.sum(dists, dim=1)

    return loss

'''
    returns frame indices where min(raw_sequence[i].z) < threshold
    assume that ground is the euclidian plane (x,y,0), transform raw_vertices before if needed
'''

def get_ground_frames(raw_sequence, threshold=0.02, device="cuda"):

    ground_indices = []
    for mesh, i in zip(raw_sequence, range(0, len(raw_sequence))):
        min = np.sort(mesh.vertices[:,2])[200]
        if (min < threshold).all():
            ground_indices.append(i)
    return torch.tensor(ground_indices).to(device)

# forward and backward feet bottom vertices, /!\ visually picked
FEET_INDICES = torch.tensor([3359, 3468, 6758, 6866]).to("cuda")
'''
    smpl_frames: frames to fit
    ground_indices: input frames where mesh should touch the ground (computed with get_ground_frames)
    returns cumulative distance to ground for each frame
    assume that ground is the euclidian plane (x,y,0)
'''
def ground_contact_distance(smpl_frames, threshold=0.05):
    # to avoid pulling only toes to the ground, but should only take feet point or it will pull all the body to the ground
    if len(smpl_frames) == 0:
        return torch.zeros((1), device=smpl_frames.device)
    dists = torch.abs(smpl_frames[:,FEET_INDICES, 2])
    dists[dists > threshold] = 0
    
    loss = torch.sum(dists, dim=1)

    return loss


# from mesh_intersection.filter_faces import FilterFaces
# import mesh_intersection.loss as collisions_loss
# '''
#     returns the interpenetration loss from torch-mesh-isect
# '''
# def smpl_interpenetration(smpl_frames:torch.Tensor, smpl_faces, bvh, pen_distance):
#     bs, nv = smpl_frames.shape[:2]
#     bs, nf = smpl_faces.shape[:2]
#     faces_idx = smpl_faces + (torch.arange(bs, dtype=torch.long).to(smpl_frames.device) * nv)[:, None, None]
#     triangles = smpl_frames.view([-1, 3])[faces_idx]
#     collision_idxs = bvh(triangles)
#     loss = pen_distance(triangles, collision_idxs).mean()
    
#     return loss
