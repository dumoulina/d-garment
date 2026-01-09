import os
import random
import json
import re
import argparse

import numpy as np
import torch
import torch.nn as nn

from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
from utils.geometry import batch_compute_points_normals, angular_distance, angular_velocity
from utils.representation import aa2six, six2aa
from utils.file_management import rename_simple
from utils.helpers import dict_to
from data.amass_dataset import AmassDataset
from tqdm.auto import tqdm

'''
Pose optimization to limit self collision -> use separate_arms from SNUG instead if you do not seek to reproduce results
Script to prepare AMASS dataset for simulation
- SMPL shape randomly sampled over an uniform distribution
'''
def main():
    description = 'Script for preparing AMASS dataset for simulating cloth over SMPL'
    parser = argparse.ArgumentParser(description=description,
                                     prog='SMPL-Prepare')
    parser.add_argument('config', type=str,
                        help='The config file')
    
    # args for the self-penetration optimization
    parser.add_argument('--log_losses', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Display the loss during the optimization' +
                        ' process')
    parser.add_argument('--optimize_pose', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable self-intersection minimization over the joint poses')
    parser.add_argument('--only_shoulder', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Only optimize the shoulder and arm joint pose of the model')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='The learning rate for Adam')
    parser.add_argument('--coll_loss_weight', default=1e-1, type=float,
                        help='The weight for the collision loss')
    parser.add_argument('--pose_reg_weight', default=1, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--speed_reg_weight', default=1e1, type=float,
                        help='The weight for the angular velocity regularizer')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--max_optim_steps', default=50, type=int,
                        help='The maximum optimization steps')

    args = parser.parse_args()

    log_losses = args.log_losses
    optimize_pose = args.optimize_pose
    only_shoulder = args.only_shoulder
    lr = args.lr
    coll_loss_weight = args.coll_loss_weight
    pose_reg_weight = args.pose_reg_weight
    speed_reg_weight = args.speed_reg_weight
    max_collisions = args.max_collisions
    max_optim_steps = args.max_optim_steps
    max_optim_steps = args.max_optim_steps

    random.seed(0)
    torch.set_float32_matmul_precision('high')
    # torch.autograd.set_detect_anomaly(True)

    with open(args.config) as f:
        config = json.load(f)

    FRAME_RATE = config["dataset"]["frame_rate"]
    old_root = config['dataset']['amass_folder']
    new_root = config['dataset']['folder']
    body_shape_sampled_scale = config['dataset']['body_shape_sampled_scale']
    body_shape_sampled_components = config['body_model']['num_betas']
    shape_random_generator = torch.Generator(device='cpu')
    collision_tolerance = config['dataset']['collision_tolerance']

    amass = AmassDataset(config)
    device = amass.device

    smpl_faces = amass.smpl_faces

    search_tree = BVH(max_collisions=max_collisions)
    pen_distance = collisions_loss.DistanceFieldPenetrationLoss()

    losses = dict()
    total_length = 0

    # preparing SMPL sequences
    for seq_idx, seq in enumerate(amass):
        print(f"{seq_idx+1}/{len(amass)}: " + seq.npz_file)

        for shape_idx in range(config['dataset']['body_shape_sample_count']):
            # if re.search("348/walking_run02_poses", seq.npz_file) is None:
            #     continue
            data = seq.sequence_data(FRAME_RATE, False)

            batch_size = len(data['poses'])
            total_length += batch_size

            # normalize trans
            data["trans"] = data["trans"] - data["trans"][0]

            # randomize shape
            random_betas = torch.rand((body_shape_sampled_components,), generator=shape_random_generator).to(device)
            random_betas = random_betas * body_shape_sampled_scale * 2 - body_shape_sampled_scale
            data['betas'] = random_betas

            # prepare optimizable variables
            to_optimize = []
            if optimize_pose:
                if only_shoulder:
                    shoulder_slice = torch.full_like(data['poses'], False, dtype=torch.bool)
                    shoulder_slice[:, 16*3:20*3, ...] = True
                    poses = data['poses'][shoulder_slice].reshape(batch_size, -1)
                else:
                    poses = data['poses']

                poses = poses.reshape(batch_size, -1, 3)
                init_pose = poses.detach().clone()
                poses = aa2six(poses)

                poses = nn.Parameter(poses)
                to_optimize.append(poses)

            optimizer = torch.optim.LBFGS(to_optimize, lr=lr, line_search_fn="strong_wolfe")

            pbar = tqdm(range(max_optim_steps))
            for step in range(max_optim_steps):
                pbar.update(1)
                def closure():
                    optimizer.zero_grad()
                    if optimize_pose:
                        aa_poses = six2aa(poses).reshape(batch_size, -1)
                        if only_shoulder:
                            data['poses'] = data['poses'].detach()#.clone()
                            data['poses'][shoulder_slice] = aa_poses.reshape(-1)
                        else:
                            data['poses'] = aa_poses

                    verts = amass.body_model(data)

                    # push vertices to add a tolerance margin
                    normals = batch_compute_points_normals(verts, smpl_faces)
                    verts += collision_tolerance * normals
                    verts[:, amass.v_mask_smpl_hand] += collision_tolerance * normals
                    triangles = verts[:, smpl_faces]
                    collision_idxs = search_tree(triangles)
                    pen = pen_distance(triangles, collision_idxs)
                    pen_loss = coll_loss_weight * pen.sum()
                    if log_losses:
                        losses['pen_loss'] = pen_loss.detach().cpu().item()

                    loss = pen_loss

                    if optimize_pose:
                        angles = aa_poses.reshape(batch_size, -1, 3)
                        reg_dist = angular_distance(angles, init_pose)
                        pose_reg_loss = pose_reg_weight * (reg_dist).sum()
                        if log_losses:
                            losses['reg_loss'] = pose_reg_loss.detach().cpu().item()
                        loss += pose_reg_loss

                        current_speed = angular_velocity(angles)
                        # current_accel = torch.diff(current_speed, dim=0).abs()
                        speed_reg_loss = speed_reg_weight * current_speed.sum()
                        if log_losses:
                            losses['accel_loss'] = speed_reg_loss.detach().cpu().item()
                        loss += speed_reg_loss

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(poses, max_norm=10.0, error_if_nonfinite=True)
                    return loss

                optimizer.step(closure)
                # scheduler.step()

                # loss_max = loss.max().detach().cpu().item()
                # loss_dist = abs(loss_max-last_loss)
                pbar.set_postfix(losses)

            pbar.close()
            
            # save fixed sequence
            path = seq.npz_file

            path = path.replace(old_root, new_root, 1)
            path = rename_simple(path)
            path = path.replace("_poses.npz", f"/shape{shape_idx}/poses.npz", 1)

            save_data = dict_to(data, 'cpu', True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **save_data)
            # torch.cuda.memory._dump_snapshot(f"./memory.pickle")
            torch.cuda.empty_cache()

    print("Total sequences: " + str(len(amass) * config['dataset']['body_shape_sample_count']))
    print("Total frames: " + str(total_length))

if __name__ == '__main__':
#     torch.cuda.memory._record_memory_history(
#        max_entries=100000
#    )
    main()
