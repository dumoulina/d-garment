import torch
import trimesh
from tqdm import tqdm
from human_body_prior.tools.omni_tools import makepath
from utils.file_management import list_files
import os
import re
from multiprocessing import Pool
from utils.helpers import alphanum_key
from pytorch3d.structures import Meshes

class PC_Sequence:
    """Class encapsulating a pointcloud sequence with timestamps"""

    def _loader(self, f):
        try:
            mesh = trimesh.load(f, process=False)
        except Exception as e:
            print(str(e) + " : " + str(f))
        return mesh
    
    def __init__(self, folder,
                 dtype=torch.float32, device="cuda",
                 timestamps=None, end=None, start=None, relative=False, ext="ply", faces=None, **kwargs):
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
        super().__init__(**kwargs)

        self.timestamps = timestamps
        self.device = device
        self.dtype = dtype
        self.faces = faces

        self.folder = folder

        files = list_files(self.folder, ext)

        files.sort(key=alphanum_key)
        
        number_of_pc = len(files)
        self.limits = torch.tensor([0, int(number_of_pc)])
            
        if start is None:
            start = self.limits[0]
        else:
            self.limits[0] = start
            i = int(re.findall(r'\d+', files[0])[-1])
            while i < self.limits[0]:
                if (int(re.findall(r'\d+', files[0])[-1]) < self.limits[0]):
                    removed_file = files.pop(0)
                    print("Not using: " + removed_file)
                i+=1
        
        if end is None:
            end = self.limits[1]
        else:
            if relative:
                self.limits[1] = self.limits[0]+end
            else:
                self.limits[1] = end+1
            i = int(re.findall(r'\d+', files[len(files)-1])[-1])
            while i >= self.limits[1]:
                if (int(re.findall(r'\d+', files[len(files)-1])[-1]) >= self.limits[1]):
                    removed_file = files.pop()
                    print("Not using: " + removed_file)
                i-=1

        # manual regex depending of file naming
        # self.folder,  "manifold_250k_%06d.%s"%(i,ext)), process=False) for i in range(self.limits[0], self.limits[1])]

        # with Pool(min(os.cpu_count(), len(files))) as p:
        #     self.meshes = p.map(self._loader, files)

        vertices = []
        list_faces = []
        for f in tqdm(files):
            mesh = self._loader(f)
            vertices.append(torch.tensor(mesh.vertices, device=device, dtype=self.dtype))

            if faces is None:
                list_faces.append(torch.tensor(mesh.faces, device=device, dtype=torch.int64))
        
        if faces is not None:
            list_faces = [faces]
    
        self.meshes = Meshes(vertices, list_faces)

        if(self.timestamps is None or len(self.timestamps)!=len(self.meshes)):
            # print("Wrong timestamps, default to a 50fps sequence")
            self.timestamps = torch.linspace(0,(self.limits[1]-self.limits[0])/50,steps=self.limits[1]-self.limits[0]).to(self.device) 

    def duration(self):
        """Returns the duration of the sequence in seconds"""
        return self.timestamps[-1].item()

    def number_of_frames(self):
        return self.limits[1]-self.limits[0]

    def frame_data(self, index, interpol_mode=0, unit="s", nopad=False):
        """Access a certain frame r set of frames of the sequence

        Args:
            index (int or float): timestamps in second or indexes of the frame to access
            interpol_mode (int, optional): _description_Interpolation stategy for inexact timestamps. Defaults to 0.
            unit (str, optional): unit of idx "s" for seconds "i" for integer indexes. Defaults to "s". 
            Requesting index in seconds for sequences with missing frames won't work
            nopad (bool, optional): If nopad is True, requesting indexes out of the sequence will raise an exception . Defaults to False.

        Raises:
            NotImplementedError: _description_
            IndexError: _description_

        Returns:
            (dict): pointcloud data
        """
        meshes = self.meshes

        if(unit == "s"):
            frame_id = self.framerate * \
                torch.tensor(index, dtype=torch.float32)
            if interpol_mode == 0:
                frame_id = frame_id.int()
            else:
                raise NotImplementedError(
                    "Interpolation mode not implemented yet")
        else:
            frame_id = torch.tensor(index, dtype=torch.int32)

        if (frame_id >= self.number_of_frames()).any():
            if(nopad):
                raise IndexError("Not enough frames")
            frame_id[frame_id >= self.number_of_frames()] = (
                self.number_of_frames()-1)

        data = {"meshes": [], "ind": [],"timestamps":[]}
        for fid in frame_id:
            data["meshes"].append(meshes[fid])
            data["ind"].append(fid)
            data["timestamps"].append(self.timestamps[fid])
        
        return data

    def save(self, new_folder):
        """Save the sequence at the location given by new_file

        Args:
            new_folder (str): Path where the data will be written
        """

        makepath(new_folder)
        with open(os.path.join(new_folder, "folder.txt"), "w+") as f:
            f.write(self.folder)

        torch.save(self.limits, os.path.join(
            new_folder, "limits.pt"))


    def get_data(self):
        return self.meshes,self.timestamps

