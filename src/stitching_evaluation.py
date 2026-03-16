import os
import random
import argparse
import numpy as np
import torch
import json
from pytorch3d.io import save_obj
from utils.uv_tools import apply_displacement
from model.VDM_latent import VDMStableDiffusionPipeline
from data.garment_dataset import GarmentDataset
from data.normalization import unnormalize_cloth
from diffusers.schedulers import DPMSolverMultistepScheduler

if __name__ == '__main__':

    description = 'Script to compute the metrics of the test dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Metrics-computation')
    parser.add_argument('config', type=str,
                        help='The config file')
    parser.add_argument('out', type=str,
                        help='meshes will be saved there')
    parser.add_argument('--subdivide', action='store_true',
                        help='Whether to use the subdivided template for evaluation')
    parser.add_argument('--step', type=int, default=20,
                        help='Number of diffusion steps to use during evaluation')
    args = parser.parse_args()

    index = 500

    with open(args.config) as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = GarmentDataset(config, device="cuda", subsets=['TEST'], subdivide=args.subdivide)
    sequences = dataset.get_seq_dict()
    template = dataset.template
    pipeline = VDMStableDiffusionPipeline.from_pretrained(config["output"]["folder"], torch_dtype=torch.float16).to("cuda:0")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")

    diffusion_steps = args.step
    faces = template.f.cpu()

    data = dataset[index]

    @torch.no_grad()
    def generate(data, interpolation=None):
        img = pipeline(data, num_inference_steps=diffusion_steps,generator=torch.Generator(device='cpu').manual_seed(0))
        data.cloth_vertices = apply_displacement(template, img, dataset.img_size, interpolation=interpolation)
        unnormalize_cloth(data)
        output = os.path.join(args.out, f"tri_{index:06d}_{str(interpolation)}.obj")
        print(output)
        os.makedirs(os.path.dirname(output), exist_ok=True)

        save_obj(output, data.cloth_vertices[0].cpu(), faces, verts_uvs=template.vt, faces_uvs=template.fvt, texture_map=img[0].abs().cpu())

    generate(data, interpolation=None)
    generate(data, interpolation='nearest')
    generate(data, interpolation='bilinear')
    generate(data, interpolation='bicubic')