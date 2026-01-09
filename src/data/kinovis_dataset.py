import os
import torch

from pytorch3d.io import load_ply
from torch.utils.data import Dataset
from utils.file_management import list_files
from utils.helpers import alphanum_key
from data.smpl_sequence import SMPL_Sequence
from body_models.smpl_model import SMPLH
from data.structures import FittingDict, LazyDict
from pytorch3d.structures import Meshes
from tqdm.auto import tqdm

class KinovisDataset(Dataset):
    '''
        Each element of the dataset consists in:
        - a cloth mesh, and its corresponding vector displacement map to the template
        - frames picked from the corresponding SMPL_Sequence
        - an ARGUS config json linking to garment template mesh and material
    '''
    @torch.no_grad()
    def __init__(self, config, dtype=torch.float32, device="cpu"):
        '''
        Args:
            config (Dict): Configuration from json file.
            dtype and device
            subsets: names of the top folders to use in config['data_dir'], single empty string means all ['']
        '''
        self.sequences = None
        self.data_dir = config["kinovis"]["folder"]
        self.seq_len = config["model"]["sequence_length"]
        self.img_size = config["model"]["image_size"]
        self.device = device
        self.trans_normalized = config["normalization"]["trans"]
        self.rot_normalized = config["normalization"]["rotation"]
        self.dtype = dtype
        self.lazy_data: list[LazyDict] = []


        config_cp = config.copy()
        config_cp["device"] = device
        self.body_model = SMPLH(config_cp)
        
        for seq_file in list_files(self.data_dir, "poses.npz"):
            directory = os.path.dirname(seq_file)
            directory = os.path.join(directory, "LOD0")
            try:
                frame_files = sorted(list_files(directory, ".ply", False), key=alphanum_key)
            except:
                continue

            for frame_idx in range(len(frame_files)):
                
                frame = LazyDict(
                    frame_path = frame_files[frame_idx],
                    frame_idx = frame_idx,
                    smpl_seq_path = seq_file
                )

                self.lazy_data.append(frame)
        self.get_seq_dict()

    @torch.no_grad()
    def __len__(self):
        return len(self.lazy_data)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> FittingDict:

        lazy_data = self.lazy_data[idx]

        smpl_seq = SMPL_Sequence(lazy_data["smpl_seq_path"], dtype=self.dtype).to(self.device)
        seq = smpl_seq.smpl_data()

        seq["poses"] = seq["poses"][:, :66]

        frame_idx = lazy_data["frame_idx"]

        last_frames = torch.arange(frame_idx-self.seq_len+1, frame_idx+1, 1, dtype=int).clamp(0, smpl_seq.number_of_frames()-1)

        v, f = load_ply(lazy_data["frame_path"])

        captured_mesh = Meshes(
            v.to(self.device, self.dtype),
            f.to(self.device)
        )

        if len(seq["betas"].shape) > 1:
            seq["betas"] = seq["betas"][frame_idx]

        frame_data = FittingDict(
            target_mesh = captured_mesh,
            betas = seq["betas"],
            poses = seq["poses"][last_frames],
            trans = seq["trans"][last_frames]
        )

        frame_data.root_joint_position = self.body_model(frame_data, Jtr=True)[1][-1][0]

        return frame_data
    
    # returns a dict containing sequence starting index and len with sequence path as key
    def get_seq_dict(self):

        if self.sequences:
            return self.sequences
        
        self.sequences = dict()
        for i, data in enumerate(self.lazy_data):
            key = os.path.dirname(data['frame_path']).removeprefix(self.data_dir).removeprefix("/").removesuffix("/LOD0")
            frame_index = data["frame_idx"]

            if key in self.sequences:
                self.sequences[key][1] = max(frame_index, self.sequences[key][1])
            else:
                self.sequences[key] = [-1, frame_index]

            if data["frame_idx"] == 0:
                self.sequences[key][0] = i

        return self.sequences
    
    def get_body_meshes(self, seq_path):

        seq = self.get_seq_dict()[seq_path]
        SEQ_START = seq[0]
        SEQ_LEN = seq[1]

        body_sequence = []

        for i in tqdm(range(SEQ_START, SEQ_START+SEQ_LEN)):
            data = self[i]

            body_vertices = self.body_model(data)[-1].cpu()
            body_sequence.append(body_vertices)

        return torch.stack(body_sequence)
