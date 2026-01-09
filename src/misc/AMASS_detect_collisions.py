import argparse
import torch
import json
import pickle

from mesh_intersection.bvh_search_tree import BVH
from utils.file_management import list_files
from utils.smpl_tools import separate_arms
from data.smpl_sequence import SMPL_Sequence
from body_models.smpl_model import SMPLH
from tqdm import tqdm

'''
Script to compute AMASS self-collision stats for each sequence used to clean the body motion dataset
Output a pickle dict file where:
    -key is a sequence path and
    -value is a tensor with the percentage of self-intersecting faces for each frame
'''
if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help='Folder containing (.npz files of AMASS) to be checked' +
                        ' for collisions')
    parser.add_argument('config', type=str,
                        help='Config file for SMPLH (.npz file)')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--log_file', default="../log/amass_intersect_log.pkl", type=str,
                        help='The maximum number of bounding box collisions')

    args, _ = parser.parse_known_args()
    max_collisions = args.max_collisions
    log_file = args.log_file
    files = list_files(args.data_path, '.npz')

    with open(args.config) as f:
        config = json.load(f)
        config["dtype"] = torch.float32
        # config["device"] = 'cpu'
        smplh = SMPLH(config)
    faces = smplh.get_faces().to(device, torch.long)
    triangle_count = float(faces.shape[0])

    m = BVH(max_collisions=max_collisions)

    result = dict()
    for f in tqdm(files):
        print(f)
        seq = SMPL_Sequence(f)
        data = dict()
        data["trans"] = seq.data["trans"].to(device, dtype=torch.float32)
        data["poses"] = seq.data["poses"][:, :66].to(device, dtype=torch.float32)
        data['poses'] = separate_arms(data['poses'], config['body_model']['arm_rotation'])
        data["betas"] = seq.data["betas"][None,...].to(device, dtype=torch.float32)
        vertices = smplh(data).to(dtype=torch.float32)
        triangles = vertices[:, faces]
        
        outputs = m(triangles)
        mask = outputs[..., 0] >= 0
        collisions = outputs[mask]
        
        count_before = torch.count_nonzero(mask, dim=-1) * (100 / triangle_count)

        result[f] = count_before.detach().cpu()
        
    with open(log_file, 'wb') as f:
        pickle.dump(result, f)
