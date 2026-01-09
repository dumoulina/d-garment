import pytorch3d.structures
import torch
import torch.nn.functional as F
import pytorch3d
from data.garment_dataset import GarmentDataset
from utils.uv_tools import apply_displacement
from pytorch3d.ops.knn import knn_points, knn_gather
from utils.geometry import angle_v2, l2dist
from pytorch3d.loss import mesh_laplacian_smoothing

def smooth_loss(vertices, faces):
    mesh = pytorch3d.structures.Meshes(vertices, faces)
    return mesh_laplacian_smoothing(mesh)

def uv_loss(prediction:torch.Tensor, dataset:GarmentDataset, predicted_vertices:torch.Tensor = None):

    batch_size = prediction.shape[0]
    position_map = dataset.template_position_map[None, ...].expand(batch_size, -1, -1, -1).to(prediction.device) + prediction

    if predicted_vertices is None:
        predicted_vertices = apply_displacement(dataset.template, prediction)

    correspondance = dataset.correspondance[None,:,None].expand(batch_size, -1, 3).to(prediction.device)
    continuous_positions = predicted_vertices.gather(1, correspondance).reshape_as(position_map)

    loss = F.mse_loss(position_map, continuous_positions, reduction='none')

    return (torch.exp(loss)-1)

'''
Reversed Ecloth from {Estimation of Human Body Shape in Motion with Wide Clothing, J. Yang}
'''
def collision_loss(cloth_vertices, body_vertices, body_normals, cloth_normals=None, eps=0.2):

    # ∑Nv δout δNN knn.dists
    knn = knn_points(cloth_vertices, body_vertices)
    dists = knn.dists.squeeze(-1) # torch.sqrt

    # gather knn values
    body_knn_normals = knn_gather(body_normals, knn.idx).squeeze(-2)
    body_knn_vertices = knn_gather(body_vertices, knn.idx).squeeze(-2)

    # compute δNN (outliers)
    dists[dists>=eps] = 0
    if cloth_normals is not None:
        angle = angle_v2(body_knn_normals, cloth_normals, return_degrees=True)
        dists[angle>=60] = 0

    # compute δout
    cloth_smpl_vector = cloth_vertices - body_knn_vertices
    angle = angle_v2(body_knn_normals, cloth_smpl_vector, return_degrees=True)

    dists[angle <= 90] = 0    # outside

    return dists

'''
Ecloth from {Estimation of Human Body Shape in Motion with Wide Clothing, J. Yang}
'''
def clothing_energy(body_vertices, gt_vertices, gt_normals, body_normals=None):

    # ∑Nv δout δNN knn.dists
    knn = knn_points(body_vertices, gt_vertices)
    dists = knn.dists

    # gather knn values
    gt_knn_normals = knn_gather(gt_normals, knn.idx).squeeze(-2)
    gt_knn_vertices = knn_gather(gt_vertices, knn.idx).squeeze(-2)

    # compute δNN (outliers)
    dists[dists>=0.2] = 0
    if body_normals is not None:
        angle = angle_v2(body_normals, gt_knn_normals, return_degrees=True)
        dists[angle>=60] = 0

    # compute δout
    gt_smpl_vector = body_vertices - gt_knn_vertices
    angle = angle_v2(gt_knn_normals, gt_smpl_vector, return_degrees=True)
    dists[angle>=90] = 0    # inside

    return dists

'''
Computes the speed loss for a sequence of vertices.
'''
def temporal_loss(sequence: torch.Tensor):
    speed_loss = l2dist(sequence[:-1], sequence[1:])
    return speed_loss

# from mesh_intersection.filter_faces import FilterFaces
# import mesh_intersection.loss as collisions_loss
# '''
#     returns the interpenetration loss from torch-mesh-isect, would be useful for cloth self collisions
# '''
# def smpl_interpenetration(smpl_frames:torch.Tensor, smpl_faces, bvh, pen_distance):
#     bs, nv = smpl_frames.shape[:2]
#     bs, nf = smpl_faces.shape[:2]
#     faces_idx = smpl_faces + (torch.arange(bs, dtype=torch.long).to(smpl_frames.device) * nv)[:, None, None]
#     triangles = smpl_frames.view([-1, 3])[faces_idx]
#     collision_idxs = bvh(triangles)
#     loss = pen_distance(triangles, collision_idxs).mean()
    
#     return loss

def kl_gaussian(pred):
    mu = pred.mean(dim=0)
    var = pred.var(dim=0)
    logvar = torch.log(var + 1e-5)
    return 0.5 * torch.sum(mu**2 + var - logvar - 1)
