import os
import json
import argparse
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.ply_io import load_ply
from pytorch3d.transforms import axis_angle_to_matrix
from data.garment_dataset import GarmentDataset
from utils.file_management import list_files, list_subfolders
from losses.metrics import compute_all, Metrics
from utils.geometry import resolve_penetration, MeshCache
from tqdm import tqdm
import random
import numpy as np
import pickle
from utils.helpers import alphanum_key

def get_gt_vertices(seq_key, step=5, normalize=True):
    seq = gt_sequences[seq_key]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    ground_truth = torch.empty(0).to(device)
    body = torch.empty(0).to(device)

    for i in range(SEQ_START, SEQ_START+SEQ_LEN, step):
        data = dataset[i]
        v = data.cloth_vertices
        ground_truth = torch.cat((ground_truth, v[None,...]))
        if normalize:
            data['trans'][...] = 0
            data['poses'][..., :3] = 0
        body_vertices = body_model(data)[-1]
        body = torch.cat((body, body_vertices[None,...]))

    return body, ground_truth

def normalize_vertices(vertices, seq_key, step=5):
    seq = gt_sequences[seq_key]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    for i, data_idx in enumerate(range(SEQ_START, SEQ_START+SEQ_LEN, step)):
        data = dataset[data_idx]
        data.cloth_vertices = vertices[i]
        vertices[i] = normalize_cloth(data)

    return vertices

def normalize_cloth(data):

    # SMPL: v = lbs(poses) + trans
    vertices = data['cloth_vertices']
    global_trans = data['trans']

    aa_rotation = data['poses'][..., :3]
    offset = data['root_joint_position']

    rotation_matrix = axis_angle_to_matrix(aa_rotation).transpose(-1, -2)

    vertices -= offset
    vertices = rotation_matrix.matmul(vertices.transpose(-1, -2)).transpose(-1, -2)
    vertices += offset

    vertices -= global_trans

    return vertices

def align_hood_vertices(vertices, seq_key, step=5):
    seq = gt_sequences[seq_key]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    offset = torch.zeros_like(vertices)
    
    for i, data_idx in enumerate(range(SEQ_START, SEQ_START+SEQ_LEN, step)):
        data = dataset[data_idx]
        offset[i] = data.root_joint_position-data.trans

    # 90 degree rotation
    vertices -= offset
    vertices[..., 2] *= -1
    vertices = vertices[..., [0, 2, 1]]
    vertices += offset

    return vertices

def load_hood_sequence(path:str, offset=0, step=1):
    with open(path, 'rb') as seq_file:
        seq_dict = pickle.load(seq_file)
    vertices = torch.Tensor(seq_dict['pred']).to(device, torch.float32)[59+offset::step]
    return vertices

def load_sequence(path:str, step=1):
    vertices = torch.load(path, weights_only=True).to(device, torch.float32)[::step]
    return vertices

def load_zhang_sequence(seq_dir:str, step=1):

    eval_dir = os.path.join(seq_dir, "evaluation")
    vertices = []

    files = list_files(eval_dir)
    files.sort(key=alphanum_key)
    for file in files:
        with open(file, 'rb') as f_ptr:
            v, _ = load_ply(f_ptr)
        vertices.append(v)
    vertices = torch.stack(vertices).to(device, torch.float32)[::step]

    return vertices

if __name__ == '__main__':

    description = 'Script to compute the metrics of the test dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Metrics-computation')
    parser.add_argument('config', type=str,
                        help='The config file')
    parser.add_argument('eval_dir', type=str,
                        help='The evaluation directory aligned with the dataset ' \
                        'with a tensor file for each sequence, metrics will be saved there')
    parser.add_argument('template_path', type=str,
                        help='The path to the template mesh file corresponding to the cloth rest shape')
    parser.add_argument('--post_process', action='store_true',
                        help='Whether to use the post-processing to resolve penetrations')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    config['normalization']['rotation'] = True
    config['normalization']['trans'] = True
    config["model"]["sequence_length"] = 1

    dataset = GarmentDataset(config, device=device, subsets=['TEST'], subdivide=False)
    body_model = dataset.body_model
    gt_sequences = dataset.get_seq_dict()
    gt_template = dataset.template
    gt_template = MeshCache(gt_template.v.unsqueeze(0), gt_template.f.unsqueeze(0))

    template = load_objs_as_meshes([args.template_path], device=device, load_textures=False)
    template = MeshCache(template.verts_padded(), template.faces_padded())
    template_faces = template.faces_packed()
    same_topology = (gt_template.faces_packed().shape[0] == template_faces.shape[0])

    if 'MGDDG' in args.eval_dir:
        files = sorted(list_subfolders(args.eval_dir, "simulation_"))
    else:
        files = sorted(list_files(args.eval_dir, "simulation_"))
    body_faces = body_model.get_faces()

    metrics = Metrics()
    total_frames = 0
    progress_bar = tqdm(total=len(files), desc='Computing metrics')
    for seq_file, gt_key in zip(files, gt_sequences):
        progress_bar.set_postfix_str(gt_key)

        body_v, gt_vertices = get_gt_vertices(gt_key, 5)

        if 'HOOD' in args.eval_dir:
            vertices = load_hood_sequence(seq_file, 0, 3)
            vertices = align_hood_vertices(vertices, gt_key)
            vertices = normalize_vertices(vertices, gt_key, 5)
        elif 'ContourCraft' in args.eval_dir:
            vertices = load_hood_sequence(seq_file, 2, 3)
            vertices = align_hood_vertices(vertices, gt_key)
            vertices = normalize_vertices(vertices, gt_key, 5)
        elif 'MGDDG' in args.eval_dir:
            vertices = load_zhang_sequence(seq_file, 5)
        else:
            vertices = load_sequence(seq_file, 5)
            vertices = normalize_vertices(vertices, gt_key, 5)

        # Clip to the minimum batch size, aligned on the ground truth
        bs = gt_vertices.shape[0]
        vertices = vertices[:bs]
        total_frames += bs

        gt_meshes = MeshCache(gt_vertices, gt_template.faces_padded().expand(bs,-1,-1))
        body_meshes = MeshCache(body_v, body_faces.unsqueeze(0).expand(bs,-1,-1))

        if args.post_process:
            body_normals = body_meshes.verts_normals_padded()
            vertices = resolve_penetration(vertices,
                                        body_v,
                                        body_normals,
                                        body_faces,
                                        min_distance=0.015,
                                        max_distance=10,
                                        copy=False)

        meshes = MeshCache(vertices, template_faces.unsqueeze(0).expand(bs,-1,-1))

        metrics += compute_all(gt_meshes, meshes, body_meshes, gt_template, template, end_device='cpu', same_topology=same_topology, z_axe=1)

        progress_bar.update(1)

    progress_bar.close()

    metrics /= total_frames
    metric_dict = metrics.item()
    metric_dict['num_frames'] = total_frames

    print(metric_dict)

    suffix = ""
    if args.post_process:
        suffix = "_post_processed"
    path = os.path.join(args.eval_dir, f"metrics{suffix}.json")

    with open(path, "w") as f:
        json.dump(metric_dict, f)
