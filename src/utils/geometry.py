import torch
import math
from utils.representation import aa2rotmat
import pytorch3d.transforms
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d import _C
from pytorch3d.ops import cot_laplacian, sample_points_from_meshes

class MeshCache(Meshes):
    ''' Helper class to store sampled points and normals for evaluation'''
    def __init__(self, verts, faces, sample=50000):
        super().__init__(verts, faces)
        self.sample = sample

        self.sampled_points, self.sampled_normals = sample_points_from_meshes(self, num_samples=sample, return_normals=True)
        self.edge_length = edge_length(self)
        self.face_area = torch.stack(torch.split(face_area(self.verts_packed(), self.faces_packed()), self.num_faces_per_mesh().tolist()))

def l2dist(a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, ord=2, dim=-1)

# from https://github.com/pytorch/pytorch/issues/59194
def angle_v2(a, b, dim=-1, return_degrees=True):
    a_norm = torch.norm(a, dim=dim, keepdim=True)
    b_norm = torch.norm(b, dim=dim, keepdim=True)
    angle = torch.atan2(
        torch.norm((a * b_norm - a_norm * b), dim=dim),
        torch.norm((a * b_norm + a_norm * b), dim=dim)
    ) * 2
    if return_degrees:
        return angle * (180/math.pi)
    return angle

def face_area(vertices:torch.Tensor, faces:torch.Tensor) -> torch.Tensor:
    tris = vertices[faces]
    a = tris[...,1, :] - tris[...,0, :]
    b = tris[...,2, :] - tris[...,0, :]
    cross = torch.linalg.cross(a, b)
    area = 0.5 * torch.linalg.vector_norm(cross, ord=2, dim=-1)
    return area

# compute the area associated to each vertex, somewhat wrong but simple to compute
def vertex_area(vertices:torch.Tensor, faces:torch.Tensor) -> torch.Tensor:
    face_areas = face_area(vertices, faces)  # (F, )
    third_area = face_areas / 3.0
    vertex_local_area = torch.zeros(vertices.shape[0], device=vertices.device)
    vertex_local_area.index_add_(0, faces[:, 0], third_area)
    vertex_local_area.index_add_(0, faces[:, 1], third_area)
    vertex_local_area.index_add_(0, faces[:, 2], third_area)
    return vertex_local_area

# from https://github.com/facebookresearch/pytorch3d/issues/736
@torch.no_grad()
def batch_compute_points_normals(points, faces):
    verts_normals = torch.zeros_like(points)
    vertices_faces = points[:, faces]
    faces_normals = torch.cross(
        vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
        vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
        dim=-1,
    )

    verts_normals.index_add_(1, faces[:, 0], faces_normals)
    verts_normals.index_add_(1, faces[:, 1], faces_normals)
    verts_normals.index_add_(1, faces[:, 2], faces_normals)

    norm = torch.norm(verts_normals, dim=-1, keepdim=True)
    verts_normals /= norm

    return verts_normals

# beware of mesh winding
def triangle_normal(triangles):
    U = triangles[..., 2] - triangles[..., 0]
    V = triangles[..., 1] - triangles[..., 0]
    normals = torch.cross(U, V, dim=-1)

    norm = torch.norm(normals, dim=-1, keepdim=True)
    normals /= norm
    
    return normals

# angularDistance = 2*acos(abs(parts(p*conj(q))));
# compute angular distance between two vectors (in axis angle (seq, ..., 3))
def angular_distance(x: torch.Tensor, y: torch.Tensor, return_cos=False, eps=1e-6) -> torch.Tensor:
    
    x_R = aa2rotmat(x)
    y_R = aa2rotmat(y)

    return pytorch3d.transforms.so3.so3_relative_angle(x_R, y_R, cos_angle=return_cos, cos_bound=eps, eps=eps)

# compute angular speed across sequence of angles (in axis angle (seq, ..., 3)), use cos instead of angle for numerical stability
def angular_velocity(aa_angles: torch.Tensor) -> torch.Tensor:
    prev = aa_angles[:-1]
    next = aa_angles[1:]
    dist = angular_distance(prev, next)
    return dist

# Moves penetrating vertices of an open surface outside a watertight mesh.
def resolve_penetration(vertices, body_vertices, body_normals, body_faces, min_distance=1e-5, max_distance=1, copy=True):

    bs = vertices.shape[0]

    # Find closest watertight surface points for each open vertex
    knn = knn_points(vertices, body_vertices)
    nn_body_normals = knn_gather(body_normals, knn.idx)[:,:,0]
    nn_body_vertices = knn_gather(body_vertices, knn.idx)[:,:,0]
    penetration_distances = torch.sqrt(knn.dists[:,:,0])

    pcls = Pointclouds(vertices)
    meshes = Meshes(body_vertices, body_faces[None,...].expand(bs,-1,-1))

    # packed representation for pointclouds
    points = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    dists, nn_faces_idxs = _C.point_face_dist_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        5e-3,
    )

    penetration_distances = dists.reshape(vertices.shape[:-1]).sqrt()

    # nn_faces_idxs = nn_faces_idxs.reshape(vertices.shape[:-1]) % body_faces.shape[0]
    # nn_faces = body_faces[nn_faces_idxs] # [212, 2636, 3]
    # batch_indices = torch.arange(bs, dtype=torch.long).view(bs, 1, 1).expand_as(nn_faces)
    # nn_body_triangles = body_vertices[batch_indices, nn_faces] # [212, 2636, 3, 3]
    # nn_body_face_normals = triangle_normal(nn_body_triangles) # [212, 2636, 3]
    
    # Identify penetrating vertices
    penetration_vectors = vertices - nn_body_vertices
    dot = (penetration_vectors * nn_body_normals).sum(dim=-1)
    inside = (dot < 0) & (penetration_distances < max_distance)
    outside_near = (dot >= 0) & (penetration_distances < min_distance)

    # Move vertices outward
    if copy:
        vertices = vertices.clone()
    displacement = nn_body_normals * (min_distance + penetration_distances[..., None])
    vertices[inside] += displacement[inside]
    displacement = nn_body_normals * (min_distance - penetration_distances[..., None])
    vertices[outside_near] += displacement[outside_near]

    return vertices

# compute vertex mean curvature using cotangent laplacien method
# no batch to avoid OOM
def mean_curvature(vertices, faces):
    L, inv_area = cot_laplacian(vertices, faces)
    laplacian = L @ vertices
    mean_curvature = 0.5 * (laplacian.norm(dim=1) * inv_area)
    return mean_curvature

# compute vertex gaussian curvature using angle defect method
# no batch to avoid OOM
def gaussian_curvature(vertices, faces) -> torch.Tensor:

    # packed faces: (F, 3), packed verts: (V, 3)
    # gather triangle vertex positions
    tris = vertices[faces]  # (F, 3, 3)

    a = tris[:, 0]
    b = tris[:, 1]
    c = tris[:, 2]

    # edge vectors for angles
    ab = b - a
    ac = c - a
    ba = a - b
    bc = c - b
    ca = a - c
    cb = b - c

    angle_a = angle_v2(ab, ac, return_degrees=False)
    angle_b = angle_v2(ba, bc, return_degrees=False)
    angle_c = angle_v2(ca, cb, return_degrees=False)

    # Sum angles per vertex
    vertex_angle_sum = torch.zeros(vertices.shape[0], device=vertices.device)
    vertex_angle_sum.index_add_(0, faces[:, 0], angle_a)
    vertex_angle_sum.index_add_(0, faces[:, 1], angle_b)
    vertex_angle_sum.index_add_(0, faces[:, 2], angle_c)

    # Compute vertex areas
    vertex_local_area = vertex_area(vertices, faces)

    # angle-defect (Gaussian) curvature per vertex
    curvature = (2.0 * torch.pi - vertex_angle_sum) / (vertex_local_area + 1e-8)

    return curvature
# D = ((w * (theta1 - theta2)**2)).sum().sqrt()
def edge_length(meshes:Meshes, stacked:bool=True) -> torch.Tensor:
    '''
    Compute edge lengths for each mesh in a batch of meshes.
    Args:
        meshes (Meshes): A batch of meshes.
    Returns:
        List of Tensors: Each tensor contains the edge lengths for the corresponding mesh.
    '''
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N
    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    edge_lengths = l2dist(v0, v1)
    edge_lengths_per_mesh = edge_lengths.split(num_edges_per_mesh.tolist())

    if stacked:
        return torch.stack(edge_lengths_per_mesh)

    return edge_lengths_per_mesh

def mean_edge_strain(meshes:MeshCache, template:MeshCache):
    mesh_edge_length = meshes.edge_length
    strain = (mesh_edge_length - template.edge_length)/template.edge_length
    strain = strain*strain*mesh_edge_length

    return strain.mean(dim=1)

def mean_area_strain(meshes:MeshCache, template:MeshCache):
    face_areas = meshes.face_area
    strain = (face_areas - template.face_area)/template.face_area
    weighted_strain = strain*strain*face_areas

    return weighted_strain.mean(dim=1)

def mesh_edge_loss(meshes:Meshes):
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N
    edges_packed_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n),)

    loss = torch.cat(edge_length(meshes, stacked=False))
    loss_per_mesh = torch.zeros(num_edges_per_mesh, dtype=loss.dtype, device=loss.device)
    loss_per_mesh.index_add_(0, edges_packed_to_mesh_idx, loss)

    return loss_per_mesh / num_edges_per_mesh

def bbox_height(vertices: torch.Tensor, dim=0, axe=2) -> torch.Tensor:
    min_z = vertices[..., axe].min(dim=dim).values
    max_z = vertices[..., axe].max(dim=dim).values
    height = max_z - min_z
    return height

def center_of_mass(meshes:Meshes) -> torch.Tensor:

    vertices = meshes.verts_packed()
    faces = meshes.faces_packed()

    face_areas = face_area(vertices, faces)
    face_centers = vertices[faces].mean(dim=1)
    weighted_face_centers = face_centers * face_areas[:, None]

    face_to_mesh_idx = meshes.faces_packed_to_mesh_idx()
    num_meshes = len(meshes)
    centers = torch.zeros((num_meshes, 3), device=vertices.device)
    centers.index_add_(0, face_to_mesh_idx, weighted_face_centers)
    total_areas = torch.zeros(num_meshes, device=vertices.device)
    total_areas.index_add_(0, face_to_mesh_idx, face_areas)
    centers = centers / total_areas[:, None]
    return centers