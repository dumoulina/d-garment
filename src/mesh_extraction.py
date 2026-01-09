import os
import json
import argparse
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes, save_ply
from pytorch3d.io.ply_io import load_ply
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes
from data.garment_dataset import GarmentDataset
from data.normalization import unnormalize_cloth
from utils.file_management import list_files, list_subfolders
from utils.geometry import resolve_penetration
from utils.visualization import Sequencevisualizer
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import pickle
from utils.helpers import alphanum_key
from body_models.smpl_model import BodyModel

def raw2trimesh(vertices, faces, rgb=[0.5,0.5,0.5]):
    return [trimesh.Trimesh(v, faces, vertex_colors=rgb, process=False) for v in vertices]

def get_gt_vertices(seq_key, step=5, normalize=False):
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

def unnormalize_cloth(data):
    # SMPL: v = lbs(poses) + trans
    vertices = data['cloth_vertices']
    global_trans = data['trans']

    aa_rotation = data['poses'][..., :3]
    offset = data['root_joint_position'] - global_trans

    rotation_matrix = axis_angle_to_matrix(aa_rotation)

    vertices -= offset
    vertices = rotation_matrix.matmul(vertices.transpose(-1, -2)).transpose(-1, -2)
    vertices += offset

    vertices += global_trans

    return vertices

def normalize_vertices(vertices, seq_key, step=5):
    seq = gt_sequences[seq_key]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    for i, data_idx in enumerate(range(SEQ_START, SEQ_START+SEQ_LEN, step)):
        data = dataset[data_idx]
        data.cloth_vertices = vertices[i]
        vertices[i] = normalize_cloth(data)

    return vertices

def unnormalize_vertices(vertices, seq_key, step=5):
    seq = gt_sequences[seq_key]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    for i, data_idx in enumerate(range(SEQ_START, SEQ_START+SEQ_LEN, step)):
        data = dataset[data_idx]
        data.cloth_vertices = vertices[i]
        vertices[i] = unnormalize_cloth(data)

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
    # equivalent:
    # theta = torch.pi / 2
    # sin = np.sin(theta)
    # cos = np.cos(theta)
    # rotation_matrix = torch.tensor([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]).to(vertices.device, vertices.dtype)  # Rx
    # vertices = rotation_matrix.matmul(vertices.transpose(-1, -2)).transpose(-1, -2)
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

    config['normalization']['rotation'] = False
    config['normalization']['trans'] = False
    config["model"]["sequence_length"] = 1

    dataset = GarmentDataset(config, device=device, subsets=['TEST'], subdivide=False)
    body_model = dataset.body_model
    gt_sequences = dataset.get_seq_dict()

    template = load_objs_as_meshes([args.template_path], device=device, load_textures=False)
    template_faces = template.faces_packed().cpu()

    if 'MGDDG' in args.eval_dir:
        files = sorted(list_subfolders(args.eval_dir, "simulation_"))
    else:
        files = sorted(list_files(args.eval_dir, "simulation_"))
    body_faces = body_model.get_faces()

    for seq_file, gt_key in zip(files, gt_sequences):

        # the frame was picked on a sequence which looks the best on the ground truth
        # e.g. the frame is looking good independently of the previous frames which can be biased with large motions
        # B16/shape0/simulation_1
        gt_key_parts = gt_key.split('/')
        sim_n = gt_key_parts[-1][-1]
        shape_n = gt_key_parts[-2][-1]
        seq_name = gt_key_parts[-3]
        def check_conditions(sim_c, shape_c, seq_filec):
            return sim_n == sim_c and shape_n == shape_c and seq_filec in seq_file

        if not (check_conditions('1', '0', 'B16') or check_conditions('0', '1', 'B17') or check_conditions('1', '1', 'C19')):
            continue
        print(seq_file)
        body_v, gt_vertices = get_gt_vertices(gt_key, 5)

        if 'HOOD' in args.eval_dir:
            vertices = load_hood_sequence(seq_file, 0, 3)
            vertices = align_hood_vertices(vertices, gt_key)
        elif 'ContourCraft' in args.eval_dir:
            vertices = load_hood_sequence(seq_file, 2, 3)
            vertices = align_hood_vertices(vertices, gt_key)
        elif 'MGDDG' in args.eval_dir:
            vertices = load_zhang_sequence(seq_file, 5)
            vertices = unnormalize_vertices(vertices, gt_key)
        else:
            vertices = load_sequence(seq_file, 5)

        # Clip to the minimum batch size, aligned on the ground truth
        bs = gt_vertices.shape[0]
        vertices = vertices[:bs]
    
        if args.post_process:
            body_meshes = Meshes(body_v, body_faces.unsqueeze(0).expand(bs,-1,-1))
            body_normals = body_meshes.verts_normals_padded()
            vertices = resolve_penetration(vertices,
                                        body_v,
                                        body_normals,
                                        body_faces,
                                        min_distance=0.015,
                                        max_distance=10,
                                        copy=False)
        
        # Debug visualization
        # gt_meshes = raw2trimesh(gt_vertices.cpu(), template_faces, [1.,0.,0.])
        # body_meshes = raw2trimesh(body_v.cpu(), body_faces.cpu(), [0.,0.,1.])
        # meshes = raw2trimesh(vertices.cpu(), template_faces, [1.,0.,0.])
        # cross = trimesh.creation.axis(origin_size=0.1)
        # for i in range(len(gt_meshes)):
        #     meshes[i] += body_meshes[i]
        # Sequencevisualizer(meshes)
        # exit()

        save_dir = os.path.join(args.eval_dir, seq_name + f"_shape{shape_n}_simulation_{sim_n}")
        os.makedirs(os.path.join(save_dir, f"mesh"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, f"body"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, f"GT"), exist_ok=True)
        for i in range(bs):
            path = os.path.join(save_dir, f"mesh/{i:03d}.ply")
            save_ply(path, vertices[i], template_faces)
            path = os.path.join(save_dir, f"GT/{i:03d}.ply")
            save_ply(path, gt_vertices[i], template_faces)
            path = os.path.join(save_dir, f"body/{i:03d}.ply")
            save_ply(path, body_v[i], body_faces)
