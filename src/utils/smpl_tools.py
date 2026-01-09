
import json

import numpy as np
import torch
import trimesh
from utils.representation import aa2quat, quat2aa, multiply_quat
from body_models.smpl_model import SMPLH

# Parse a given file to a (1, 8) float numpy array
def parse(path):
    with open(path) as f:
        txt = f.read()
    return np.array([txt.split(" ")[0:8]]).astype(float)

# segment SMPL faces
def segment_SMPL(faces:torch.Tensor, smpl_seg_file:str) -> torch.Tensor:
    """Function to get relevant smpl faces for the cloth simulation"""

    with open(smpl_seg_file) as f:
        smpl_seg = json.load(f)
                
    parts = torch.Tensor([5517, 5683, 5750, 2058, 2289, 2222])
    parts = torch.cat((parts, torch.arange(6159, 6231), torch.arange(2698, 2770)))

    for k in ['rightToeBase', 'rightFoot', 'leftToeBase', 'leftFoot', 'rightHandIndex1', 'leftHandIndex1']:
        parts = torch.cat((parts, torch.Tensor(smpl_seg[k])))

    parts = parts.to(faces.device)
    mask = torch.isin(faces, parts, invert=True).all(dim=-1)

    return faces[mask]

def mask_SMPL_hand(smpl_seg_file:str, smpl_v_count=6890) -> torch.Tensor:
    """Function to get hand smpl vertices"""

    with open(smpl_seg_file) as f:
        smpl_seg = json.load(f)

    hand_parts = []

    # Add segmented parts (if present in JSON)
    for k in ['rightHandIndex1', 'leftHandIndex1', 'leftHand', 'rightHand']:
        if k in smpl_seg:
            hand_parts += smpl_seg[k]
    # Add hardcoded
    hand_parts += [5517, 5683, 5750, 2058, 2289, 2222]
    hand_parts += list(range(6159, 6231))
    hand_parts += list(range(2698, 2770))

    mask = torch.ones(smpl_v_count, dtype=torch.bool)
    hand_parts = torch.tensor(hand_parts, dtype=torch.long).unique()
    mask[hand_parts] = False

    return mask

def get_body_meshes(data: dict, model: SMPLH) -> list[trimesh.Trimesh]:
    vertices = model(data)
    faces = model.get_faces()

    return [trimesh.Trimesh(v.cpu(), faces.cpu()) for v in vertices]

def separate_arms(poses: torch.Tensor, angle=20, left_arm=17, right_arm=16):
    """
    Modify the SMPL poses to avoid self-intersections of the arms and the body.
    Adapted from the code of SNUG (https://github.com/isantesteban/snug/blob/main/snug_utils.py#L93)

    :param poses: [Nx72] SMPL poses
    :param angle: angle in degree to rotate the arms
    :param left_arm: index of the left arm in the SMPL model
    :param right_arm: index of the right arm in the SMPL model
    :return:
    """
    num_joints = poses.shape[-1] // 3
    angle *= torch.pi/180

    poses = poses.reshape((-1, num_joints, 3))

    rot = aa2quat(torch.tensor([0, 0, -angle], device=poses.device))[None,...].expand(poses.shape[0], -1)
    poses[:, left_arm] = quat2aa(multiply_quat(rot, aa2quat(poses[:, left_arm])))

    rot = aa2quat(torch.tensor([0, 0, angle], device=poses.device))[None,...].expand(poses.shape[0], -1)
    poses[:, right_arm] = quat2aa(multiply_quat(rot, aa2quat(poses[:, right_arm])))

    return poses.reshape((poses.shape[0], -1))
