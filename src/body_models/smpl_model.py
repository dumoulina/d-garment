from human_body_prior.body_model.body_model import BodyModel
import torch.nn as nn
import torch

SMPLH_JOINTS_BODY = 22
SMPLH_JOINTS_HAND = 30

class SMPLH(nn.Module):
    """ SMPLH body model based on the body model BodyModel implementation of human body prior"""

    def __init__(self, configuration):
        """
        Args:
            configuration (dict): Gives information on the location of the body_model information
        """
        super(SMPLH, self).__init__()

        self.num_betas = configuration["body_model"]["num_betas"]
        self.device = configuration["device"]
        self.body_model = BodyModel(configuration["body_model"]["smpl_info"],
                                    num_betas=self.num_betas).to(self.device)

    def forward(self, data, joints=list(range(SMPLH_JOINTS_BODY)), use_hands=False, fist_hand=False, Jtr=False):
        """Process the batch of smpl data and ouputs a batch of meshes
        Args:
            data ([dict]): dictionnary representing a batch of smpl data with keys ["poses", "trans", "betas"]
            joints ([list], optional): Subset of joints to consider, missing joint rotation set to I. Defaults to list(range(SMPLH_JOINTS_BODY)).
            use_hands (bool, optional): Whether or not the hand pose is given in the data. Defaults to False.
            fist_hand (bool, optional): Whether or not the hand pose will form a fist and the wrist will be set to 0. Defaults to False.
            Jtr (bool, optional): Whether or not joints positions will be returned. Defaults to False.
        Returns:
            [tensor]: The SMPL meshes computed from the input data BSx6890x3
            if Jtr is True:
            tuple[tensor, tensor]: The SMPL meshes computed from the input data BSx6890x3, the corresponding joint positions BSx52x3
        """

        data2 = dict()
        for k in ["poses", "trans", "betas"]:
            if(len(data[k].shape) == 1):  # if single frame and no batch
                data2[k] = data[k][None,...]
            else:
                data2[k] = data[k]

        poses = data2["poses"]

        dim = 3
        if(use_hands):
            pose_hand = poses[:, SMPLH_JOINTS_BODY*dim:]
        else:
            pose_hand = torch.zeros(
                (poses.shape[0], SMPLH_JOINTS_HAND*dim), dtype=data2["poses"].dtype, device=self.device)

        if fist_hand:
            # set wrists to zero
            with torch.no_grad():
                poses[..., 20*dim:22*dim] = 0
            # fist ~~ remove fingers by posing them into the hand
            # left fist
            pose_hand[..., 2:20:9] = -torch.pi
            pose_hand[..., 19] = -torch.pi
            pose_hand[..., 28] = -torch.pi
            pose_hand[..., 37] = torch.pi*.2
            pose_hand[..., 40] = torch.pi*.3
            pose_hand[..., 43] = torch.pi*.3

            # right fist
            pose_hand[..., 46:64:9] = -torch.pi
            pose_hand[..., 64] = -torch.pi
            pose_hand[..., 73] = -torch.pi
            pose_hand[..., 82] = -torch.pi*.2
            pose_hand[..., 85] = -torch.pi*.3
            pose_hand[..., 88] = -torch.pi*.3

        full_body_pose = torch.zeros(
            (poses.shape[0], SMPLH_JOINTS_BODY,dim), dtype=data2["poses"].dtype, device=self.device)
        poses_rs = poses.view(poses.shape[0],-1,dim)

        full_body_pose[:,joints,:]=poses_rs
        full_body_pose = full_body_pose.reshape(poses.shape[0],-1)

        model = self.body_model(
            root_orient=full_body_pose[:, 0:dim],
            pose_body=full_body_pose[:, dim:SMPLH_JOINTS_BODY*dim],
            pose_hand=pose_hand,
            betas=data2["betas"][:, :self.num_betas],
            trans=data2["trans"])

        if(Jtr):
            return model.v.reshape(-1, 6890, 3), model.Jtr.reshape(-1, 52, 3)
        else:
            return model.v.reshape(-1, 6890, 3)

    def get_faces(self):
        return self.body_model.f

    def to(self, device):
        self.device = device
        self.body_model = self.body_model.to(device)
        return self
