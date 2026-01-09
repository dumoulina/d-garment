import numpy as np
import torch

class SMPL_Sequence:
    """Class encapsulating a Npz file which represents a sequence"""

    def __init__(self, npz_file, load_smpl=True, dtype=torch.float32, device="cuda",
                 pose_hand=False, joints=list(range(22)), **kwargs):
        """Initialize a sequence
        Args:
            npz_file (str): The path to the npz file where the data is located
            load_smpl (bool, optional): If True, load the data in device. Defaults to True.
            dtype (optional): Data type of the sequence. Defaults to torch.float64.
            device (str, optional): Defaults to "cuda".
            pose_hand (bool, optional): If True, the hand data is in the smpl data. Defaults to False.
        Raises:
            KeyError: Raised when failing validity check
        """
        super().__init__(**kwargs)

        self.npz_file = npz_file
        self.data = None
        self.dtype = dtype
        self.device = device
        self.pose_hand = pose_hand
        self.joints = joints
    
        if(load_smpl):
            with np.load(npz_file) as data:
               
                self.data = {k: torch.from_numpy(
                    data[k]).to(device, dtype) for k in ["poses", "trans", "mocap_framerate", "betas"]}
               
                self.framerate_ = data["mocap_framerate"].item()
                self.number_of_frames_ = data["poses"].shape[0]
                try:
                    self.gender = data["gender"]
                except:
                    self.gender = "neutral"
            if not self.pose_hand:
                self.data['poses'] = self.data['poses'][..., :66]

    def smpl_data(self):
        """Access the smpl parameters of the sequence
        Returns:
            dict: The smpl parameters of the sequence
        """
        if(self.data is None):
            data_np = np.load(self.npz_file)  
            try:
                data = {k: torch.from_numpy(
                    data_np[k]).type(self.dtype).to(self.device) for k in ["poses", "trans", "mocap_framerate", "betas", "gender"]}
            except:
                print(list(data.keys()), self.npz_file)
                exit(0)
            if not self.pose_hand:
                self.data['poses'] = self.data['poses'][..., :66]
        else:
            data = self.data

        return data.copy()

    def get_body_vertices(self, body_model=None):
        """Access the vertices representing the sequence
        Args:
            body_model (optional): Model used to extrapolate a mesh from the smpl data. Defaults to None.
        Returns:
            tensor: A MxNx3 tensor with M the number of frames and N the number of vertices per mesh or None
        """

        if(body_model is None):
            with np.load(self.npz_file) as data:
                try:
                    return torch.from_numpy(data["mesh_data"]).type(self.dtype).to(self.device)
                except:
                    return None
        else:
            # return body_model(self.smpl_data(), use_hands=self.pose_hand).type(self.dtype).to(self.device)
            return body_model(self.smpl_data(),
                                use_hands=self.pose_hand,
                                joints=self.joints).type(self.dtype).to(self.device)

    def duration(self):
        """Returns the duration of the sequence in seconds"""
        return self.number_of_frames_/self.framerate_

    def framerate(self):
        return self.framerate_

    def number_of_frames(self):
        return self.number_of_frames_

    def to(self, device):
        """Transfer the sequence to a new device
        Args:
            device (str)
        Returns:
            SMPL_sequence: the casted sequence
        """

        self.device = device
        if(self.data is not None):
            for k in self.data.keys():
                try:
                    self.data[k] = self.data[k].to(self.device)
                except:
                    pass

        return self

    def cast(self, new_type):
        """Cast the sequence to a new type
        Args:
            new_type : the new data format
        Returns:
            SMPL_sequence: the casted sequence
        """

        self.type = new_type
        if(self.data is not None):
            for k in self.data.keys():
                try:
                    self.data[k] = self.data[k].type(self.type)
                except:
                    pass

        return self

    def frame_data(self, index:torch.Tensor, interpol_mode=0, unit="s", pad=False):
        """Access frame data specified by index
        Args:
            index (list/int): Indexes of the desired frames
            interpol_mode (int, optional): How to interpol for unexisting timestamps. Defaults to 0.
            unit (str, optional): If s, the index will be interpreted as seconds, otherwise as frame indices. Defaults to "s".
            pad (bool, optional): If True, returns the nearest frame for any index outside the sequence range, else remove them. Defaults to False.
        Returns:
            dict: SMPL data for desired frames
        """

        data = self.smpl_data()

        frame_id_base = index.detach().clone()


        if unit == "s":
            frame_id = self.framerate() * frame_id_base
            dim_size = frame_id[-1].clone()
        else:
            dim_size = data["poses"].shape[0] - 1
            frame_id = frame_id_base

        if pad:
            frame_id[frame_id > dim_size] = dim_size
            frame_id[frame_id < 0] = 0
        else:
            frame_id = frame_id[frame_id < dim_size]
            frame_id = frame_id[frame_id >= 0]
        
        data_frame = dict()

        if interpol_mode == 0:      # no interpolation
            frame_id = frame_id.floor().long()
            data_frame = {k: data[k][frame_id.tolist(), :] for k in ["poses", "trans"]}
        elif interpol_mode == 1:    # linear interpolation
            for k in ["poses", "trans"]:
                start = data[k][frame_id.floor().long()]
                end = data[k][(frame_id.ceil().long())]
                inter_val = frame_id.frac()[..., None].expand(-1, start.shape[-1]).detach().clone()
                data_frame[k] = torch.lerp(start, end, inter_val)
                
        data_frame["betas"] = data["betas"].detach().clone()
        data_frame["timing"] = frame_id_base.tolist()
        data_frame["gender"] = self.gender

        return data_frame
    
    def sequence_data(self, out_fps, pad=False):
        """Return sequence at a desired frame rate
        Args:
            out_fps (int): The desired frame rate
        Returns:
            dict: SMPL data sequence interpoled at desired framerate
        """

        frames = torch.arange(0, self.duration(), 1/out_fps, device=self.device)
        data_seq = self.frame_data(frames, interpol_mode=1, unit="s", pad=pad)
        data_seq["mocap_framerate"] = out_fps
        return data_seq
    
    def save(self, new_file):
        """Save the sequence at the location given by new_file
        Args:
            new_file (str): Path where the data will be written
        """

        data = {
                "poses":self.data["poses"].detach().cpu().numpy(),
                "trans":self.data["trans"].detach().cpu().numpy(),
                "betas":self.data["betas"].detach().cpu().numpy(),
                "mocap_framerate":self.framerate_,
                "gender": self.gender
                }
        np.savez(new_file, **data)
