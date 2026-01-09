#

import os
import random
import json
import re
import argparse
import torch

from utils.file_management import list_files
from utils.geometry import resolve_penetration, batch_compute_points_normals
from data.amass_dataset import AmassDataset
from tqdm.auto import tqdm
from pytorch3d.io import load_ply, save_ply
'''
Script to fix cloth intersecting SMPL in dataset
'''
def main():
    description = 'Script to fix cloth intersecting SMPL in dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch-fix')
    parser.add_argument('config', type=str,
                        help='The config file')

    args = parser.parse_args()

    random.seed(0)

    with open(args.config) as f:
        config = json.load(f)

    dataset_path = config['dataset']['folder']
    collision_tolerance = config["dataset"]["collision_tolerance"]

    dataset = AmassDataset(config, dataset_path)
    dataset.body_model.requires_grad_(False)
    dataset.body_model.eval()
    body_faces = dataset.smpl_faces.detach()

    pbar = tqdm(range(len(dataset)))

    for seq_idx, seq in enumerate(dataset):
        # if re.search("B16", seq.npz_file) is None:
        #     continue
        # print(seq.npz_file)
        seq_path = os.path.dirname(seq.npz_file)

        body_verts = dataset.body_model(seq.data, fist_hand=True)
        body_normals = batch_compute_points_normals(body_verts, body_faces)

        simulations = list_files(seq_path, ".json")
        for simulation in simulations:
            files = list_files(os.path.dirname(simulation), ".ply")
            files.sort()

            cloth_verts = torch.empty(0)
            for i, f in enumerate(files):
                cloth_verts = torch.cat((cloth_verts, load_ply(f)[0][None,...]))

            cloth_verts = cloth_verts.to(body_verts.device, torch.float)

            cloth_verts = resolve_penetration(cloth_verts, body_verts, body_normals, body_faces, collision_tolerance, copy=False)

            for v, f in zip(cloth_verts, files):
                save_ply(f, v)

            pbar.update(1)

        torch.cuda.empty_cache()

    pbar.close()

if __name__ == '__main__':
    main()
