import numpy as np
import torch
import trimesh
import json
from utils.file_management import list_files
from data.smpl_sequence import SMPL_Sequence
from data.pc_sequence import PC_Sequence
from utils.visualization import Sequencevisualizer
from torch.utils.data import Dataset
from data.structures import FittingDict, Meshes
from utils.geometry import batch_compute_points_normals
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
# from pytorch3d.structures import Meshes

class Fitting_Sequence(PC_Sequence, SMPL_Sequence, Dataset):
    """Class encapsulating a smpl and point cloud sequence to fit"""
    
    def __init__(self, folder, config, body_model=None,
                 dtype=torch.float32, device="cuda",
                 timestamps=None, start=0, end=None, ext="ply", faces=None, num_samples=1000):
        """Initialize a sequence

        Args:
            folder (str): path to the pointclouds files, pointcloud name format is "model-%05d.%s"%(i,ext)
            dtype: Defaults to torch.float32.
            device: Defaults to "cuda".
            timestamps : If None sequence is assumed to be a complete 50fps sequence. 
            Otherwise it should be a tensor of timestamp
            start (int): starting frame
            end (int) : Last model to consider, If relative is true, end=start+end
            and a frame interval in limits.pt

        Raises:
            KeyError: [description]
        """
        self.seq_len = config['model']['sequence_length']
        npz_file = list_files(folder, "poses.npz")[0]
        self.body_model = body_model
        super().__init__(folder=folder, npz_file=npz_file, dtype=dtype, device=device,
                         timestamps=timestamps, ext=ext, faces=faces)

        while self.data['betas'].dim()>1:
            self.data['betas'] = self.data['betas'][0]

        if end is None:
            end = len(self.meshes)

        self.trimeshes = None
        self.meshes = self.meshes[start:end]
        self.data["poses"] = self.data["poses"][start:end]
        self.data["trans"] = self.data["trans"][start:end]
        self.number_of_frames_ = self.data["poses"].shape[0]
        self.update_smpl_meshes()
        self.num_samples = num_samples
        self.sample()

    def sample(self):
        self.sampled_vertices = []
        self.sampled_normals = []
        for mesh in self.meshes:
            v = mesh.verts_packed()[None, ...]
            normals = mesh.verts_normals_packed()
            vertices, selected = sample_farthest_points(v, K=self.num_samples)
            self.sampled_vertices.append(vertices[0])
            self.sampled_normals.append(normals[selected[0]])

        self.sampled_vertices = torch.stack(self.sampled_vertices)
        self.sampled_normals = torch.stack(self.sampled_normals)

    def get_trimeshes(self):
        if self.trimeshes is None:
            self.trimeshes = []
            for v, f in zip(self.meshes.verts_list(), self.meshes.faces_list()):
                trim = trimesh.Trimesh(v.detach().cpu(), f.detach().cpu())
                self.trimeshes.append(trim)
        return self.trimeshes

    def visualize(self):
        colored = self.get_trimeshes().copy()
        for mesh in colored:
            rgb = np.zeros_like(mesh.vertices)
            rgb[..., 2] = 1
            mesh.visual.vertex_colors = rgb

        if self.body_model is not None:
            faces = self.body_model.get_faces().cpu()
            vertices = self.get_body_vertices(self.body_model).detach().cpu()
            
            rgb = np.zeros_like(vertices.squeeze())[0]
            rgb[..., 0] = 1
            for i, v in enumerate(vertices):
                colored[i] += trimesh.Trimesh(v, faces, vertex_colors=rgb)

        Sequencevisualizer(colored)

    def save_smpl(self, path):
        data = {
            "poses":self.data["poses"].detach().cpu().numpy(),
            "trans":self.data["trans"].detach().cpu().numpy(),
            "betas":self.data["betas"].detach().cpu().numpy(),
            "mocap_framerate":50
            }

        np.savez(path, **data)

    def update_smpl_meshes(self):
        self.body_vertices, root_joint_position = self.body_model(self.smpl_data(), Jtr=True)
        self.root_joint_position = root_joint_position[:, :1]
        self.body_normals = batch_compute_points_normals(self.body_vertices, self.body_model.get_faces())

    def remove_capture_outliers(self, smpl_seg_file:str, threshold = 1e-3):

        with open(smpl_seg_file) as f:
            smpl_seg = json.load(f)

        outlier_parts = []

        # Add segmented parts (if present in JSON)
        for k in ['head', 'neck', 'rightHandIndex1', 'leftHandIndex1', 'leftHand', 'rightHand', 'leftFoot', 'leftLeg', 'leftToeBase', 'rightFoot', 'rightLeg', 'rightToeBase']:
            if k in smpl_seg:
                outlier_parts += smpl_seg[k]
        # Add hardcoded
        outlier_parts += [5517, 5683, 5750, 2058, 2289, 2222]
        outlier_parts += list(range(6159, 6231))
        outlier_parts += list(range(2698, 2770))

        smpl_mask = torch.zeros(self.body_vertices.shape[-2], dtype=torch.bool)
        outlier_parts = torch.tensor(outlier_parts, dtype=torch.long).unique()
        smpl_mask[outlier_parts] = True

        # compute threshold
        body_vertices = self.body_vertices
        outlier_vertices = body_vertices[..., smpl_mask, :]

        updated_vertices = []
        updated_faces = []
        for vertices, faces, body_vertices in zip(self.meshes.verts_list(), self.meshes.faces_list(), self.body_vertices):

            outlier_vertices = body_vertices[..., smpl_mask, :]

            mesh = trimesh.Trimesh(vertices.cpu(), faces.cpu())

            knn = knn_points(vertices[None,...], outlier_vertices[None,...])
            dists = knn.dists.squeeze()

            vertex_mask =  dists > threshold
            face_mask = vertex_mask[faces].all(-1)
            face_indices = torch.nonzero(face_mask).squeeze()

            threshold_mesh = mesh.submesh([face_indices.cpu()], append=True)

            components = threshold_mesh.split(only_watertight=False)
            cleaned_mesh = max(components, key=lambda m: len(m.faces))

            updated_vertices.append(torch.tensor(cleaned_mesh.vertices, device=self.device, dtype=self.dtype))
            updated_faces.append(torch.tensor(cleaned_mesh.faces, device=self.device, dtype=torch.int64))

        self.meshes = Meshes(updated_vertices, updated_faces)
        self.sample()

        self.trimeshes = None

    def frame_data(self):
        pass

    def __len__(self):
        return self.number_of_frames_

    def __getitem__(self, idx: int) -> FittingDict:
        last_frames = torch.arange(idx-self.seq_len+1, idx+1, 1, dtype=int).clamp(0, self.number_of_frames()-1)
        smpl_data = self.data
        # mesh = self.meshes[idx]
        # if self.faces is not None:
        #     faces = self.faces
        # else:
        #     faces = torch.tensor(mesh.faces)
        frame_data = FittingDict(
            target_mesh = self.meshes[idx], # Meshes([torch.tensor(mesh.vertices, dtype=self.dtype)], [faces])
            sampled_vertices = self.sampled_vertices[idx],
            sampled_normals = self.sampled_normals[idx],
            betas = smpl_data["betas"],
            poses = smpl_data["poses"][last_frames],
            trans = smpl_data["trans"][last_frames],
            root_joint_position = self.root_joint_position[idx],
            body_vertices = self.body_vertices[idx],
            body_normals = self.body_normals[idx],
            index = idx
        )

        return frame_data
