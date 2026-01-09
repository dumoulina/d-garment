#
import os
import json
import argparse
import torch
from data.garment_dataset import GarmentDataset
from tqdm.auto import tqdm
import trimesh

'''
Script to compute the mean shape geometry of the dataset
'''
def main():
    torch.multiprocessing.set_start_method('spawn')

    description = 'Script to compute the mean shape geometry of the dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Mean-shape')
    parser.add_argument('config', type=str,
                        help='The config file')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    dataset = GarmentDataset(config, device='cuda')    
    sum = torch.zeros_like(dataset.template.v)

    pbar = tqdm(range(len(dataset)))
    for data in dataset:
        sum += data.cloth_vertices
        pbar.update(1)

    pbar.close()
    mesh = trimesh.Trimesh(sum.cpu()/len(dataset), dataset.template.f.cpu())
    mesh.export(os.path.join(dataset.data_dir, 'mean_shape.obj'))

if __name__ == '__main__':
    main()
