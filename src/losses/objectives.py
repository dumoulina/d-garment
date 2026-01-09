# Copyright 2024 adakri
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes, Pointclouds


# Copied from pytorch3d.loss.point_mesh_distance
# Change 1: return per-point and per-face distances (s2m and m2s)
# Change 2: scale points and mesh verts by 100. since there may be a problem when values are too small
#   during check in pytorch3d/pytorch3d/csrc/utils/geometry_utils.h IsInsideTriangle BaryCentricCoords3Forward
#   kEpsilon = 1e-8 is applied but this gets too big if distances between verts are very small.
def py3d_point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds, do_f2p=True):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)
    scale_fac = 100. # see explanation above

    # packed representation for pointclouds
    points = pcls.points_packed() * scale_fac # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed() * scale_fac
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    # point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    # num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    # weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # weights_p = 1.0 / weights_p.float()
    # point_to_face = point_to_face * weights_p / (scale_fac ** 2)
    point_to_face = point_to_face / (scale_fac ** 2)
    #point_dist = point_to_face.sum() / N

    if do_f2p:
        # face to point distance: shape (T,)
        face_to_point = face_point_distance(
            points, points_first_idx, tris, tris_first_idx, max_tris
        )

        # weight each example by the inverse of number of faces in the example
        # tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
        # num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
        # weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
        # weights_t = 1.0 / weights_t.float()
        # face_to_point = face_to_point * weights_t / (scale_fac ** 2)
        face_to_point = face_to_point / (scale_fac ** 2)
        #face_dist = face_to_point.sum() / N
    else:
        face_to_point = None

    return point_to_face, face_to_point#, p2f_inds, f2p_inds #point_dist + face_dist


# from pytorch3d.loss.chamfer.py
# now returns pointwise distances instead of summed/averaged value
def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """

    import torch
    import torch.nn.functional as F
    from pytorch3d.ops.knn import knn_gather, knn_points
    from pytorch3d.loss.chamfer import _handle_pointcloud_input

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    
    dist_x = torch.sqrt(x_nn.dists[..., 0])  # (N, P1)
    dist_y = torch.sqrt(y_nn.dists[..., 0])  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    return cham_x, cham_y, cham_norm_x, cham_norm_y, dist_x, dist_y


#===============================================================
# Priors, robustness and regularizers


# German macclure function
# GMoF from PROX
class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual * residual
        dist = torch.div(squared_res, squared_res + self.rho*self.rho)
        return self.rho*self.rho * dist

class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual * residual
        dist = torch.div(squared_res, squared_res + self.rho*self.rho)
        return dist

def poseprior_obj(prior_pose, model_pose_opt, pose_weights=None):
    # first 3 parameters are global rotation! <-- NOT ANYMORE
    if pose_weights is None:
        return (model_pose_opt - prior_pose).pow(2).sum()
    else:
        return (pose_weights * (model_pose_opt - prior_pose)).pow(2).sum()


def auto_regularizer(betas_opt):
    return betas_opt.pow(2).sum()