import torch
import numpy as np
from pytorch3d.loss import chamfer_distance
import trimesh
from losses.losses import collision_loss
from utils.geometry import mean_curvature, bbox_height, l2dist, center_of_mass, mean_area_strain, mean_edge_strain, MeshCache
from data.structures import TensorDict
from dataclasses import dataclass

# all metrics are summed over batches
@dataclass
class Metrics(TensorDict):
    v2v_dist: float = 0.0
    chamfer: float = 0.0
    normal: float = 0.0
    penetration: float = 0.0
    curvature: float = 0.0
    wrinkling: float = 0.0
    edge_strain: float = 0.0
    area_strain: float = 0.0
    height_diff: float = 0.0
    mass_center: float = 0.0
    mass_center_height: float = 0.0

def v2v_distance(gt_meshes:MeshCache, meshes:MeshCache) -> torch.Tensor:

    dist = l2dist(gt_meshes.verts_padded(), meshes.verts_padded()).mean(dim=-1)

    return dist.sum()

def penetration(meshes:MeshCache, body_meshes:MeshCache) -> torch.Tensor:

    pen_loss = collision_loss(meshes.sampled_points,
                              body_meshes.sampled_points,
                              body_meshes.sampled_normals,
                              eps=10.0)
    percent_intersect = float(pen_loss.count_nonzero()) / float(meshes.sample)

    return torch.tensor(percent_intersect)

def curvature(gt_meshes:MeshCache, meshes:MeshCache):

    vertices = meshes.verts_list()
    faces = meshes.faces_list()
    curv = []
    for v, f in zip(vertices, faces):
        curv.append(mean_curvature(v, f).mean())
    curv = torch.stack(curv)

    gt_vertices = gt_meshes.verts_list()
    gt_faces = gt_meshes.faces_list()
    gt_curv = []
    for v, f in zip(gt_vertices, gt_faces):
        gt_curv.append(mean_curvature(v, f).mean())
    gt_curv = torch.stack(gt_curv)

    return torch.abs(curv - gt_curv).sum()

def signed_adjacency_angle(mesh:trimesh.Trimesh):
# from trimesh.Trimesh.integral_mean_curvature
    # assign signs based on convex adjacency of face pairs
    signs = np.array([-1.0, 1.0])[mesh.face_adjacency_convex.astype(np.int64)]
    # adjust face adjacency angles with signs to reflect orientation
    return mesh.face_adjacency_angles * signs

def wrinkling(gt_meshes:MeshCache, meshes:MeshCache):

    vertices = meshes.verts_padded().cpu()
    faces = meshes.faces_list()[0].cpu()

    gt_vertices = gt_meshes.verts_padded().cpu()
    gt_faces = gt_meshes.faces_list()[0].cpu()
    meshes.faces_packed_to_edges_packed()
    curvatures = []
    for v, gt_v in zip(vertices, gt_vertices):
        mesh = trimesh.Trimesh(gt_v, gt_faces)
        gt_curv = torch.tensor(signed_adjacency_angle(mesh))
        mesh = trimesh.Trimesh(v, faces)
        curv = torch.tensor(signed_adjacency_angle(mesh))
        dist = ((curv-gt_curv)**2).mean().sqrt()
        curvatures.append(dist)

    return torch.stack(curvatures).sum()

def area_strain(gt_meshes:MeshCache, meshes:MeshCache, gt_template, template):

    m_integral_area_strain = mean_area_strain(meshes, template)
    gt_integral_area_strain = mean_area_strain(gt_meshes, gt_template)

    return abs(m_integral_area_strain - gt_integral_area_strain).sum()

def edge_strain(gt_meshes:MeshCache, meshes:MeshCache, gt_template, template):

    m_integral_edge_strain = mean_edge_strain(meshes, template)
    gt_integral_edge_strain = mean_edge_strain(gt_meshes, gt_template)

    return abs(m_integral_edge_strain - gt_integral_edge_strain).sum()

def height_diff(gt_meshes:MeshCache, meshes:MeshCache, axe=2) -> torch.Tensor:
    gt_vertices = gt_meshes.verts_padded()
    vertices = meshes.verts_padded()
    gt_height = bbox_height(gt_vertices, dim=1, axe=axe)
    height = bbox_height(vertices, dim=1, axe=axe)
    return abs(gt_height - height).sum()

def chamfer(gt_meshes:MeshCache, meshes:MeshCache) -> torch.Tensor:

    vertices, normals = meshes.sampled_points, meshes.sampled_normals
    gt_vertices, gt_normals = gt_meshes.sampled_points, gt_meshes.sampled_normals

    return chamfer_distance(vertices, gt_vertices,
                            x_normals=normals, y_normals=gt_normals,
                            batch_reduction='sum')

def mass(gt_meshes:MeshCache, meshes:MeshCache, axe=None) -> torch.Tensor:
    mass_center = center_of_mass(meshes)
    gt_mass_center = center_of_mass(gt_meshes)
    diff = l2dist(gt_mass_center, mass_center).sum()
    if axe is not None:
        return diff, abs(gt_mass_center[:, axe] - mass_center[:, axe]).sum()
    return diff

def compute_all(gt_meshes:MeshCache, meshes:MeshCache, body_meshes:MeshCache, gt_template:MeshCache, template:MeshCache, end_device='cpu', same_topology=True, z_axe=2) -> Metrics:
    v2v_distance_val = -1.0
    tri_curv = -1.0
    if same_topology:
        v2v_distance_val = v2v_distance(gt_meshes, meshes)
        tri_curv = wrinkling(gt_meshes, meshes)
    chamf, normal = chamfer(gt_meshes, meshes)
    pen = penetration(meshes, body_meshes)
    curv = curvature(gt_meshes, meshes)
    edge_strain_val = edge_strain(gt_meshes, meshes, gt_template, template)
    area_strain_val = area_strain(gt_meshes, meshes, gt_template, template)
    hdiff = height_diff(gt_meshes, meshes, axe=z_axe)
    mc, mcz = mass(gt_meshes, meshes, axe=z_axe)

    return Metrics(
            v2v_dist= v2v_distance_val,
            chamfer=chamf,
            normal=normal,
            penetration=pen,
            curvature=curv,
            wrinkling=tri_curv,
            edge_strain=edge_strain_val,
            area_strain=area_strain_val,
            height_diff=hdiff,
            mass_center=mc,
            mass_center_height=mcz,
            ).to(end_device)
