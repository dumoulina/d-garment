import os
import json
import torch
from torch.utils.data import Dataset
from utils.file_management import list_files, list_folders
from data.smpl_sequence import SMPL_Sequence
from body_models.smpl_model import SMPLH
from pytorch3d.io import load_ply
from data.structures import Mesh, LazyDict, GarmentDict
from data.normalization import normalize_cloth, unnormalize_cloth
from utils.uv_tools import mesh_uv_position


class GarmentDataset(Dataset):
    '''
        Each element of the dataset consists in:
        - a cloth mesh, and its corresponding vector displacement map to the template
        - frames picked from the corresponding SMPL_Sequence
        - a simulator json file linking to cloth material
    '''
    @torch.no_grad()
    def __init__(self, config, dtype=torch.float32, device="cpu", subsets=['ACCAD', 'KIT'], subdivide=False):
        '''
        Args:
            config (Dict): Configuration from json file.
            dtype and device
            subsets: names of the top folders to use in config['data_dir'], single empty string means all ['']
        '''
        self.sequences = None
        self.data_dir = config["dataset"]["folder"]
        self.seq_len = config["model"]["sequence_length"]
        self.img_size = config["model"]["image_size"]
        self.trans_normalized = config["normalization"]["trans"]
        self.rot_normalized = config["normalization"]["rotation"]
        self.device = device
        self.dtype = dtype
        self.lazy_data: list[LazyDict] = []
        self.subdivide = subdivide

        self.template = Mesh(config=config, dtype=dtype, device=device, subdivide=subdivide)

        config_cp = config.copy()
        config_cp["device"] = device
        self.body_model = SMPLH(config_cp)

        for subset in subsets:
            for seq_file in sorted(list_files(os.path.join(self.data_dir, subset), "poses.npz")):
                for simulation_dir in sorted(list_folders(os.path.dirname(seq_file))):
                    cloth_files = sorted(list_files(simulation_dir, ".ply"))
                    try:
                        with open(list_files(simulation_dir, ".json")[0], "r") as f:
                            simulation_config = json.load(f)

                        for frame_idx in range(len(cloth_files)):
                            frame = LazyDict(
                                cloth_path = cloth_files[frame_idx],
                                frame_idx = frame_idx,
                                bending = simulation_config["Meshes"][0]["Bending"],
                                stretching = simulation_config["Meshes"][0]["Stretching"],
                                density = simulation_config["Meshes"][0]["Area Density"],
                                smpl_seq_path = seq_file
                            )
                            self.lazy_data.append(frame)
                    except Exception as e:
                        print(str(e) + ": " + simulation_dir)

    @torch.no_grad()
    def __len__(self):
        return len(self.lazy_data)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> GarmentDict:

        lazy_data = self.lazy_data[idx]

        smpl_seq = SMPL_Sequence(lazy_data.smpl_seq_path, dtype=self.dtype).to(self.device)
        seq = smpl_seq.smpl_data()

        seq["poses"] = seq["poses"][:, :66]

        frame_idx = lazy_data.frame_idx

        last_frames = torch.arange(frame_idx-self.seq_len+1, frame_idx+1, 1, dtype=int).clamp(0, smpl_seq.number_of_frames()-1)

        vertices, _ = load_ply(lazy_data.cloth_path)
        vertices = vertices.to(self.device, self.dtype)

        frame_data = GarmentDict(
            betas = seq["betas"],
            poses = seq["poses"][last_frames],
            trans = seq["trans"][last_frames],
            bending = torch.tensor([lazy_data.bending], device=self.device),
            stretching = torch.tensor([lazy_data.stretching], device=self.device),
            density = torch.tensor([lazy_data.density], device=self.device),
            cloth_vertices = vertices
        )

        frame_data.root_joint_position = self.body_model(frame_data, Jtr=True)[1][-1][0]

        normalize_cloth(frame_data, self.trans_normalized, self.rot_normalized)
        if not self.subdivide:  # useless to compute that outside training when using subdivided template
            cloth_position_map = mesh_uv_position(frame_data.cloth_vertices, self.template.f, self.template.uv_face, self.template.uv_barycentric, self.img_size)
            vdm = cloth_position_map - self.template.position_map
            frame_data.vdm = vdm

        return frame_data
    
    # returns a dict containing sequence starting index and len with sequence path as key
    def get_seq_dict(self):

        if self.sequences:
            return self.sequences
        
        self.sequences = dict()
        for i, data in enumerate(self.lazy_data):
            key = os.path.dirname(data.cloth_path).removeprefix(self.data_dir).removeprefix("/").removesuffix("/data0/output")
            frame_index = data.frame_idx

            if key in self.sequences:
                self.sequences[key][1] = max(frame_index, self.sequences[key][1])
            else:
                self.sequences[key] = [-1, frame_index]

            if data.frame_idx == 0:
                self.sequences[key][0] = i

        return self.sequences

    def get_body_meshes(self, seq_path):

        seq = self.get_seq_dict()[seq_path]
        SEQ_START = seq[0]
        SEQ_LEN = seq[1]

        body_sequence = []

        for i in range(SEQ_START, SEQ_START+SEQ_LEN):
            data = self[i]

            body_vertices = self.body_model(data)[-1].cpu()
            body_sequence.append(body_vertices)

        return torch.stack(body_sequence)
    
    def get_gt_meshes(self, seq_path):

        seq = self.get_seq_dict()[seq_path]
        SEQ_START = seq[0]
        SEQ_LEN = seq[1]

        ground_truth = []

        for i in range(SEQ_START, SEQ_START+SEQ_LEN):
            data = self[i]

            unnormalize_cloth(data)
            ground_truth.append(data.cloth_vertices.cpu())

        return torch.stack(ground_truth)
