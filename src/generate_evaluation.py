import os
import random
import argparse
import numpy as np
import torch
import json
from utils.uv_tools import apply_displacement
from model.VDM_latent import VDMStableDiffusionPipeline
from data.garment_dataset import GarmentDataset
from data.normalization import unnormalize_cloth
from diffusers.schedulers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm

@torch.no_grad()
def generate(seq_path, diffusion_steps=20, use_material=True):

    seq = sequences[seq_path]
    SEQ_START = seq[0]
    SEQ_LEN = seq[1]

    generated = []

    pipeline.set_progress_bar_config(disable=True)
    for i in tqdm(range(SEQ_START, SEQ_START+SEQ_LEN)):
        data = dataset[i]
        img = pipeline(data, dataset.template, dataset.body_model, num_inference_steps=diffusion_steps, use_material=use_material,generator=torch.Generator(device='cpu').manual_seed(0))

        data.cloth_vertices = apply_displacement(dataset.template, img, dataset.img_size)
 
        unnormalize_cloth(data)
        assert(not torch.isnan(data.cloth_vertices).any())
        generated.append(data.cloth_vertices[0].detach())

    pipeline.set_progress_bar_config(disable=False)
    return torch.stack(generated)

if __name__ == '__main__':

    description = 'Script to compute the metrics of the test dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Metrics-computation')
    parser.add_argument('config', type=str,
                        help='The config file')
    parser.add_argument('eval_dir', type=str,
                        help='The evaluation directory aligned with the dataset ' \
                        'with a tensor file for each sequence, metrics will be saved there')
    parser.add_argument('--subdivide', action='store_true',
                        help='Whether to use the subdivided template for evaluation')
    parser.add_argument('--step', type=int, default=20,
                        help='Number of diffusion steps to use during evaluation')
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

    dataset = GarmentDataset(config, device="cuda", subsets=['TEST'], subdivide=args.subdivide)
    sequences = dataset.get_seq_dict()

    pipeline = VDMStableDiffusionPipeline.from_pretrained(config["output"]["folder"], torch_dtype=torch.float16).to("cuda:0")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")

    step = args.step
    VAL_DIR = args.eval_dir

    pipeline.compile()
    for seq_key in sequences:
        output = os.path.join(VAL_DIR, seq_key+".pt")
        print(output)
        generated = generate(seq_key, diffusion_steps=step, use_material=config["model"]["use_material"])

        os.makedirs(os.path.dirname(output), exist_ok=True)
        torch.save(generated.detach().cpu(), output)