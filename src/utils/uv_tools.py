import torch
from pytorch3d.ops.knn import knn_points
import torch.nn.functional as F

def __cross(x, y):
    return x[...,0]*y[...,1] - y[...,0]*x[...,1]

# Compute barycentric coordinates, input of same shape [..., 3], returns output of same shape
def __get_barycentric(p, triangles):

    v0 = triangles[...,1,:] - triangles[...,0,:]
    v1 = triangles[...,2,:] - triangles[...,0,:]
    v2 = p - triangles[...,0,:]

    denom = __cross(v0, v1)

    v = __cross(v2, v1) / denom
    w = __cross(v0, v2) / denom
    u = torch.ones_like(v) - v - w
    
    return torch.cat((u[..., None],v[..., None],w[..., None]), dim=-1)

# for newer version of pytorch (that are still not supported by pytorch3d) vmap might be fixed or can be used with to torch.nonzero_static
# modify chunk_size accordingly to your memory
def __find_barycentric_coords(points: torch.Tensor, triangles: torch.Tensor, chunk_size: int = 1000):

    num_points = points.shape[0]

    barycentric_coords = torch.zeros((num_points, 3),  dtype=torch.float32, device=points.device)
    triangle_indices = torch.full((num_points, ), -1, dtype=torch.long, device=points.device)

    for i in range(0, num_points, chunk_size): # cannot use vmap because of nonzero
        end = min(i + chunk_size, num_points)
        
        barycentric = __get_barycentric(points[i:end][:,None,...], triangles)

        inside = (barycentric >= 0).all(-1)   # if face mismatch artifacts use a negative epsilon value instead of zero
        valid = torch.nonzero(inside)
        valid_offset = valid[:,0] + i

        triangle_indices[valid_offset] = valid[:,1]
        barycentric_coords[valid_offset] = barycentric[valid[:,0], valid[:,1]]

    return triangle_indices, barycentric_coords

# looking for nearest face center, very fast but will output vertices that are outside corresponding triangle
def mesh_uv_barycentric_fast(vt, fvt, img_size, margin=None):

    pixel_half_size = 0.5
    
    uv_triangles = vt[fvt] * img_size # Per face uv coords
    device = uv_triangles.device

    # create a flattened grid of pixels of given img_size
    line = torch.arange(img_size, device=device)
    uv_map = torch.meshgrid(line,
                            line,
                            indexing='ij')
    pixels_flat = torch.cat(tuple(torch.dstack([uv_map[0], uv_map[1]]))) + pixel_half_size

    # Find nearest neightbours, pixel centers to triangle centers in template uv space. Mesh should be Delaunay to avoid face mismatch
    tris_centers = torch.sum(uv_triangles, dim=1).unsqueeze(0)/3.
    dists, uv_face, _ = knn_points(
        pixels_flat[None,...], 
        tris_centers,
        norm=2,
        K=1,
        version=-1)

    uv_face = uv_face.squeeze()

    mask = torch.ones_like(uv_face)
    if margin is not None and margin > 0:
        dists = dists.squeeze()
        mask[dists > margin] *= 0
    mask = mask.reshape(img_size, img_size, 1)

    # Map the 3D position to image (compute barycentric coordinates)
    # For each face the uv coordiantes
    pixels_tri_uv = uv_triangles[uv_face]
    
    barycentric = __get_barycentric(pixels_flat, pixels_tri_uv)[...,None].expand(-1,3,3)

    return uv_face, barycentric, mask

# accurate version to find barycentric coordinate but slower
def mesh_uv_barycentric(vt, fvt, img_size, pixel_offset = 0.5, margin=None):

    # Per face uv coords
    uv_triangles = vt[fvt] * img_size

    # create a flattened grid of pixels of given img_size
    line = torch.arange(img_size, device=uv_triangles.device)
    uv_map = torch.meshgrid(line,
                            line,
                            indexing='ij')
    pixels_flat = torch.cat(tuple(torch.dstack([uv_map[0], uv_map[1]]))) + pixel_offset

    uv_face, barycentric = __find_barycentric_coords(pixels_flat, uv_triangles)

    mask = (uv_face >= 0)

    pixels_to_fill = pixels_flat[~mask]
    tris_centers = torch.sum(uv_triangles, dim=1).unsqueeze(0)/3.
    dists, uv_face_fill, _ = knn_points(
    pixels_to_fill[None,...], 
    tris_centers,
    norm=2,
    K=1,
    version=-1)
    
    uv_face_fill = uv_face_fill.squeeze()
    uv_face[~mask] = uv_face_fill

    pixels_tri_uv_fill = uv_triangles[uv_face][~mask]
    filled_bary = __get_barycentric(pixels_to_fill, pixels_tri_uv_fill)
    barycentric[~mask] = filled_bary
    barycentric = barycentric[...,None].expand(-1,3,3)

    if margin is not None and margin > 0:
        dists = dists.squeeze().sqrt()
        mask[dists <= margin] = True

    mask = mask.reshape(img_size, img_size)
    
    return uv_face, barycentric, mask

# compute uv position map given precomputer mesh_uv_barycentric
def mesh_uv_position(vertices, faces, uv_face, barycentric, img_size):
    face_positions = vertices[faces]
    position_map = torch.sum(
        face_positions[uv_face] * barycentric, dim=1)
    
    position_map = position_map.reshape(img_size, img_size,3)
    return position_map

# compute texture vertex corresponding vertex and count
# def uv_to_vertex_mapping_old(vertices, faces, texture_coord, texture_faces):
#     uv_to_vertex = -torch.ones(texture_coord.shape[0], dtype=torch.long, device=texture_coord.device)
#     occurrence = torch.zeros(vertices.shape[0], dtype=torch.long, device=texture_coord.device)

#     for f, fvt in zip(faces, texture_faces):
#         for v, vt in zip(f, fvt):
#             if uv_to_vertex[vt]!= v:
#                 occurrence[v] += 1
#             uv_to_vertex[vt] = v
#     return uv_to_vertex, occurrence

# compute texture vertex corresponding vertex and count
def uv_to_vertex_mapping(vertices, faces, texture_coord, texture_faces):
    uv_to_vertex = -torch.ones(texture_coord.shape[0], dtype=torch.long, device=texture_coord.device)

    f = faces.view(-1)
    fvt = texture_faces.view(-1)
    uv_to_vertex[fvt] = f

    occurrence = torch.bincount(uv_to_vertex, minlength=vertices.shape[0])

    return uv_to_vertex, occurrence

def __unique_idx(x, dim=0):
    _, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return index

# returns new vertex positions
def apply_displacement(mesh_template, uv_displacement:torch.Tensor, img_size=None, interpolation='bilinear'):
    batched = False
    if uv_displacement.dim() == 3:
        uv_displacement = uv_displacement[None, ...]
        batched = True
    dtype = uv_displacement.dtype
    device = uv_displacement.device
    bs = uv_displacement.shape[0]

    if img_size is None:
        img_size = uv_displacement.shape[-2]

    if interpolation is None:
        f = mesh_template.f.flatten()
        fvt = mesh_template.fvt.flatten()

        v_idx = __unique_idx(f)
        uv_vertices = mesh_template.vt[fvt[v_idx]] # Per face unique uv coords

        aligned_vertices = mesh_template.v[f[v_idx]][None, ...].expand(uv_displacement.shape[0], -1, -1) # Per face unique 3d coords

        uv_vertices = (uv_vertices * img_size).to(dtype=torch.long)
        uv_vertices = uv_vertices.clamp(0, img_size-1)
        v_displacement = uv_displacement[:, uv_vertices[:,0], uv_vertices[:,1]]
    else:
        uv_vertices = mesh_template.vt.to(device, dtype)
        aligned_vertices = mesh_template.v[None, ...].expand(bs, -1, -1).to(device, dtype) # Per face unique 3d coords

        uv_vertices = uv_vertices * 2 - 1
        uv_displacement = uv_displacement.permute(0, 3, 2, 1)
        uv_vertices = uv_vertices[None, None, ...].expand(bs, -1, -1, -1)

        sampled = F.grid_sample(uv_displacement,
                                uv_vertices,
                                mode=interpolation,
                                padding_mode='border',
                                align_corners=False)
        vt_displacement = sampled.squeeze(2).permute(0, 2, 1)

        cum_displacement = torch.zeros_like(aligned_vertices)
        cum_displacement.index_add_(1, mesh_template.vt2v.to(device), vt_displacement)
        v_displacement = cum_displacement / mesh_template.vt_occurrence[None, :, None].to(device)

    new_positions = aligned_vertices + v_displacement
    if batched:
        new_positions = new_positions.squeeze(0)
    return new_positions

def pixel_vertices_correspondance(barycentric, uv_face, faces):

    nearest_simplex = torch.argmax(barycentric, dim=1)
    uv_face_vertices = faces[uv_face]
    correspondance = uv_face_vertices.gather(1, nearest_simplex)[:,0]
    
    return correspondance