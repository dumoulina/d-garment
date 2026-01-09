import torch
from torch.utils.data import Dataset
from utils.file_management import list_files, rename_simple
from data.smpl_sequence import SMPL_Sequence
from body_models.smpl_model import SMPLH
from tqdm.auto import tqdm
from utils.smpl_tools import segment_SMPL, mask_SMPL_hand

class AmassDataset(Dataset):

    def __init__(self, configuration, data_path=None, dtype=torch.float32, device=None):
        """Create a dataset encapsulating the sequences in smpl_sequences

        Args:
            configuration (dict): Provides setup information
        """
        configuration["dtype"] = dtype
        
        self.data_path = data_path
        if data_path is None:
            self.data_path = configuration["dataset"]["amass_folder"]
        

        if device:
            self.device = device
        else:
            self.device = configuration["device"]

        self.body_model = SMPLH(configuration).to(self.device).requires_grad_(False).eval()
        self.smpl_faces = self.body_model.get_faces()

        self.filtered_smpl_faces = segment_SMPL(self.smpl_faces, configuration["body_model"]["smpl_seg_file"])
        self.v_mask_smpl_hand = mask_SMPL_hand(configuration["body_model"]["smpl_seg_file"]).to(self.device)

        self.smpl_dataset = []
        files = list_files(self.data_path, pattern="poses.npz")

        files.sort(key=rename_simple)
        
        for file in tqdm(files):
            seq = SMPL_Sequence(file, device=self.device)
            seq.data['poses'] = seq.data['poses'][..., :66]
            self.smpl_dataset.append(seq)

    def __len__(self):
        return len(self.smpl_dataset)

    def __getitem__(self, idx) -> SMPL_Sequence:
        """Access the dataset

        Args:
            idx (int): idx of the element in the dataset

        Returns:
            SMPL_Sequence: class containing the smpl information and the timestamps
        """
        
        return self.smpl_dataset[idx]