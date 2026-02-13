import os
import random
import json
import argparse
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import numpy as np
import torch
import trimesh
from tqdm.auto import tqdm

from utils.file_management import rename_simple, list_files
from utils.helpers import dict_to, interpolate
from data.amass_dataset import AmassDataset

'''
Script for simulating clothes over SMPL in AMASS dataset
- several body shapes for each motion
- several cloth materials for each body shape
'''
def main():
    description = 'Script for simulating cloth over dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch-simulation')
    parser.add_argument('config', type=str,
                        help='The config file')
    parser.add_argument('--system_tmp', default=None, type=str,
                        help='The folder to use for temporary files')
    parser.add_argument('--max_workers', default=2, type=int,
                        help='The maximum used threads')

    args = parser.parse_args()
    system_tmp = args.system_tmp
    max_workers = args.max_workers

    if system_tmp is not None:
        os.makedirs(system_tmp, exist_ok=True)

    random.seed(0)

    with open(args.config) as f:
        config = json.load(f)

    FRAME_RATE = config["dataset"]["frame_rate"]
    INTERPOLATION_STEPS = config["dataset"]["interpolation_steps"]
    STABILIZATION_STEPS = config["dataset"]["stabilization_steps"]
    dataset_path = os.path.join(config['dataset']['folder'], 'SPLIT')
    amass_dataset_path = os.path.join(config['dataset']['amass_folder'], 'TEST')
    simulator_exe_path = config['dataset']['simulator_exe_path']
    material_count = config['dataset']['material_sample_count']
    shape_count = config['dataset']['body_shape_sample_count']
    body_shape_sampled_scale = config['dataset']['body_shape_sampled_scale']
    body_shape_sampled_components = config['body_model']['num_betas']
    shape_random_generator = torch.Generator(device='cpu').manual_seed(0)

    material_bounds = config["dataset"]["material"]
    dataset = AmassDataset(config, dataset_path, device='cpu')
    dataset.body_model.to('cuda')
    dataset.body_model.requires_grad_(False)
    dataset.body_model.eval()
    body_faces = dataset.smpl_faces.detach().cpu()

    print("Preparing simulation")
    pbar = tqdm(range(len(dataset)*9))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    running_simulation = set()
    tmp_dirs = {}

    # betas
    for seq_idx, seq in enumerate(dataset):
        for shape_idx in range(shape_count):
            seq_path = seq.npz_file
            out_path = seq_path.replace('/SPLIT/', '/BETAS_SPLIT/', 1)
            out_path = os.path.dirname(out_path)
            out_path = os.path.join(out_path, f"shape_{shape_idx}")

            pbar.set_postfix_str(out_path)

            data = seq.smpl_data()
            seq_len = data['trans'].shape[0]

            # randomize shape
            random_betas = torch.rand((body_shape_sampled_components,), generator=shape_random_generator)
            random_betas = random_betas * body_shape_sampled_scale * 2 - body_shape_sampled_scale
            data['betas'] = random_betas

            # UNCOMMENT to avoid body self-intersections
            # data['poses'] = separate_arms(data['poses'], config['body_model']['arm_rotation'])

            # save modified body parameters
            os.makedirs(out_path, exist_ok=True)
            np.savez(os.path.join(out_path, "poses.npz"), **data)
            data = dict_to(data, 'cuda')

            #gather canonical parameters
            canonical_betas = torch.zeros_like(data['betas'])
            canonical_trans = torch.zeros_like(data['trans'][0])
            canonical_poses = torch.zeros_like(data['poses'][0])
            canonical_poses[0] = torch.pi * 0.5

            # interpolate linearly
            interpolated_betas = interpolate(canonical_betas, data['betas'], INTERPOLATION_STEPS)
            interpolated_trans = interpolate(canonical_trans, data['trans'][0], INTERPOLATION_STEPS) # useless but why not
            interpolated_poses = interpolate(canonical_poses, data['poses'][0], INTERPOLATION_STEPS)

            # rest frames to stabilize
            interpolated_betas = torch.cat((interpolated_betas, data['betas'].expand((STABILIZATION_STEPS,-1))))
            interpolated_trans = torch.cat((interpolated_trans, data['trans'][0].expand((STABILIZATION_STEPS,-1))))
            interpolated_poses = torch.cat((interpolated_poses, data['poses'][0].expand((STABILIZATION_STEPS,-1))))

            # concat
            data['betas'] = torch.cat((interpolated_betas, data['betas'].expand(seq_len, -1)))
            data['trans'] = torch.cat((interpolated_trans, data['trans']))
            data['poses'] = torch.cat((interpolated_poses, data['poses']))

            body_verts = dataset.body_model(data).cpu()
            body_meshes = [trimesh.Trimesh(v, body_faces) for v in body_verts]

            # export body seq in temp folder for simulator (will not be cleaned if canceled!)
            tmp_dir = TemporaryDirectory(dir=system_tmp, delete=False)
            tmp_dirs[tmp_dir] = []
            for i, m in enumerate(body_meshes):
                path = os.path.join(tmp_dir.name, "mesh_%.04d.ply"%i)
                m.export(path)

            simulation_config_path = list_files(os.path.dirname(seq_path), ".json")[0]
            print(simulation_config_path)
            with open(simulation_config_path, "r") as f:
                sim_config = json.load(f)

            sim_config["Obstacles"][0]["Obj Prefix"] = os.path.join(tmp_dir.name, "mesh_")
            sim_config["Obstacles"][0]["Number Frame"] = len(body_meshes)
            sim_config["Parameters"]["Frame number"] = len(body_meshes)
            sim_config["Parameters"]["Start from Frame"] = STABILIZATION_STEPS+INTERPOLATION_STEPS
            sim_config["Obstacles"][0]["Obj Extension"] = ".ply"

            simulation_config_path = os.path.join(tmp_dir.name, "simulation_0.json")
            with open(simulation_config_path, "w") as f:
                json.dump(sim_config, f, indent=1)

            out_log = open(os.path.join(out_path, "out.log"), "w")
            err_log = open(os.path.join(out_path, "err.log"), "w")
            future = executor.submit(subprocess.run,
                                        [simulator_exe_path, simulation_config_path, out_path],
                                        stdout = out_log,
                                        stderr = err_log,
                                        check=True)
            running_simulation.add(future)
            tmp_dirs[tmp_dir].append(future)
                
            torch.cuda.empty_cache()

    # motion
    motion_test_dataset = AmassDataset(config, amass_dataset_path, device='cpu')
    for seq_idx, seq in enumerate(dataset):
        for new_motion in motion_test_dataset:
            seq_path = seq.npz_file
            out_path = seq_path.replace('/SPLIT/', '/MOTION_SPLIT/', 1)
            out_path = os.path.dirname(out_path)
            motion_name = os.path.basename(new_motion.npz_file)
            motion_name = rename_simple(motion_name)
            out_path = os.path.join(out_path, motion_name)

            pbar.set_postfix_str(out_path)

            data = seq.smpl_data()
            new_data = new_motion.sequence_data(FRAME_RATE, False)
            data['poses'] = new_data['poses']
            data['trans'] = new_data['trans']

            # remove hands
            data['poses'] = data['poses'][:, :66]

            # align initial translation to 0
            data['trans'] = data['trans'] - data['trans'][0]

            seq_len = data['trans'].shape[0]

            # UNCOMMENT to avoid body self-intersections
            # data['poses'] = separate_arms(data['poses'], config['body_model']['arm_rotation'])

            # save modified body parameters
            os.makedirs(out_path, exist_ok=True)
            np.savez(os.path.join(out_path, "poses.npz"), **data)
            data = dict_to(data, 'cuda')

            #gather canonical parameters
            canonical_betas = torch.zeros_like(data['betas'])
            canonical_trans = torch.zeros_like(data['trans'][0])
            canonical_poses = torch.zeros_like(data['poses'][0])
            canonical_poses[0] = torch.pi * 0.5

            # interpolate linearly
            interpolated_betas = interpolate(canonical_betas, data['betas'], INTERPOLATION_STEPS)
            interpolated_trans = interpolate(canonical_trans, data['trans'][0], INTERPOLATION_STEPS) # useless but why not
            interpolated_poses = interpolate(canonical_poses, data['poses'][0], INTERPOLATION_STEPS)

            # rest frames to stabilize
            interpolated_betas = torch.cat((interpolated_betas, data['betas'].expand((STABILIZATION_STEPS,-1))))
            interpolated_trans = torch.cat((interpolated_trans, data['trans'][0].expand((STABILIZATION_STEPS,-1))))
            interpolated_poses = torch.cat((interpolated_poses, data['poses'][0].expand((STABILIZATION_STEPS,-1))))

            # concat
            data['betas'] = torch.cat((interpolated_betas, data['betas'].expand(seq_len, -1)))
            data['trans'] = torch.cat((interpolated_trans, data['trans']))
            data['poses'] = torch.cat((interpolated_poses, data['poses']))

            body_verts = dataset.body_model(data).cpu()
            body_meshes = [trimesh.Trimesh(v, body_faces) for v in body_verts]

            # export body seq in temp folder for simulator (will not be cleaned if canceled!)
            tmp_dir = TemporaryDirectory(dir=system_tmp, delete=False)
            tmp_dirs[tmp_dir] = []
            for i, m in enumerate(body_meshes):
                path = os.path.join(tmp_dir.name, "mesh_%.04d.ply"%i)
                m.export(path)

            simulation_config_path = list_files(os.path.dirname(seq_path), ".json")[0]
            print(simulation_config_path)
            with open(simulation_config_path, "r") as f:
                sim_config = json.load(f)

            sim_config["Obstacles"][0]["Obj Prefix"] = os.path.join(tmp_dir.name, "mesh_")
            sim_config["Obstacles"][0]["Number Frame"] = len(body_meshes)
            sim_config["Parameters"]["Frame number"] = len(body_meshes)
            sim_config["Parameters"]["Start from Frame"] = STABILIZATION_STEPS+INTERPOLATION_STEPS
            sim_config["Obstacles"][0]["Obj Extension"] = ".ply"

            simulation_config_path = os.path.join(tmp_dir.name, "simulation_0.json")
            with open(simulation_config_path, "w") as f:
                json.dump(sim_config, f, indent=1)

            out_log = open(os.path.join(out_path, "out.log"), "w")
            err_log = open(os.path.join(out_path, "err.log"), "w")
            future = executor.submit(subprocess.run,
                                        [simulator_exe_path, simulation_config_path, out_path],
                                        stdout = out_log,
                                        stderr = err_log,
                                        check=True)
            running_simulation.add(future)
            tmp_dirs[tmp_dir].append(future)
                
            torch.cuda.empty_cache()

    # material
    for seq_idx, seq in enumerate(dataset):
        seq_path = seq.npz_file
        out_path = seq_path.replace('/SPLIT/', '/MATERIAL_SPLIT/', 1)
        out_path = os.path.dirname(out_path)

        pbar.set_postfix_str(out_path)

        data = seq.smpl_data()
        seq_len = data['trans'].shape[0]

        # UNCOMMENT to avoid body self-intersections
        # data['poses'] = separate_arms(data['poses'], config['body_model']['arm_rotation'])

        # save modified body parameters
        os.makedirs(out_path, exist_ok=True)
        np.savez(os.path.join(out_path, "poses.npz"), **data)
        data = dict_to(data, 'cuda')

        #gather canonical parameters
        canonical_betas = torch.zeros_like(data['betas'])
        canonical_trans = torch.zeros_like(data['trans'][0])
        canonical_poses = torch.zeros_like(data['poses'][0])
        canonical_poses[0] = torch.pi * 0.5

        # interpolate linearly
        interpolated_betas = interpolate(canonical_betas, data['betas'], INTERPOLATION_STEPS)
        interpolated_trans = interpolate(canonical_trans, data['trans'][0], INTERPOLATION_STEPS) # useless but why not
        interpolated_poses = interpolate(canonical_poses, data['poses'][0], INTERPOLATION_STEPS)

        # rest frames to stabilize
        interpolated_betas = torch.cat((interpolated_betas, data['betas'].expand((STABILIZATION_STEPS,-1))))
        interpolated_trans = torch.cat((interpolated_trans, data['trans'][0].expand((STABILIZATION_STEPS,-1))))
        interpolated_poses = torch.cat((interpolated_poses, data['poses'][0].expand((STABILIZATION_STEPS,-1))))

        # concat
        data['betas'] = torch.cat((interpolated_betas, data['betas'].expand(seq_len, -1)))
        data['trans'] = torch.cat((interpolated_trans, data['trans']))
        data['poses'] = torch.cat((interpolated_poses, data['poses']))

        body_verts = dataset.body_model(data).cpu()
        body_meshes = [trimesh.Trimesh(v, body_faces) for v in body_verts]

        # export body seq in temp folder for simulator (will not be cleaned if canceled!)
        tmp_dir = TemporaryDirectory(dir=system_tmp, delete=False)
        tmp_dirs[tmp_dir] = []
        for i, m in enumerate(body_meshes):
            path = os.path.join(tmp_dir.name, "mesh_%.04d.ply"%i)
            m.export(path)

        simulation_config_path = list_files(os.path.dirname(seq_path), ".json")[0]
        with open(simulation_config_path, "r") as f:
            sim_config = json.load(f)

        sim_config["Obstacles"][0]["Obj Prefix"] = os.path.join(tmp_dir.name, "mesh_")
        sim_config["Obstacles"][0]["Number Frame"] = len(body_meshes)
        sim_config["Parameters"]["Frame number"] = len(body_meshes)
        sim_config["Parameters"]["Start from Frame"] = STABILIZATION_STEPS+INTERPOLATION_STEPS
        sim_config["Obstacles"][0]["Obj Extension"] = ".ply"

        for material_idx in range(material_count):

            # sim_config["Meshes"][0]["Bending"] = 10 ** random.uniform(material_bounds["log_bending_coeff_min"], material_bounds["log_bending_coeff_max"]) # better log distribution

            sim_config["Meshes"][0]["Bending"] = random.uniform(material_bounds["bending_coeff_min"], material_bounds["bending_coeff_max"]) # original version but materials are mostly stiff
            sim_config["Meshes"][0]["Stretching"] = random.uniform(material_bounds["stretching_coeff_min"], material_bounds["stretching_coeff_max"])
            sim_config["Meshes"][0]["Area Density"] = random.uniform(material_bounds["density_min"], material_bounds["density_max"])

            simulation_config_path = os.path.join(tmp_dir.name, "simulation_" + str(material_idx) + ".json")
            with open(simulation_config_path, "w") as f:
                json.dump(sim_config, f, indent=1)

            output = os.path.join(out_path, "simulation_" + str(material_idx))
            os.makedirs(output)
            out_log = open(os.path.join(output, "out.log"), "w")
            err_log = open(os.path.join(output, "err.log"), "w")
            future = executor.submit(subprocess.run,
                                        [simulator_exe_path, simulation_config_path, output],
                                        stdout = out_log,
                                        stderr = err_log,
                                        check=True)
            running_simulation.add(future)
            tmp_dirs[tmp_dir].append(future)

        torch.cuda.empty_cache()

    pbar.close()
    print("Waiting simulations to end")
    pbar = tqdm(range(len(dataset)*9))
    for _ in as_completed(running_simulation):
        pbar.update(1)
    pbar.close()
    for td, futures in tmp_dirs.items():
        td.cleanup()

if __name__ == '__main__':
    main()
