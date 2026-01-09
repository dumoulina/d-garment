import os
import tempfile
from PIL import Image
import trimesh
import pyrender
from multiprocessing import Pool
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
import pyglet

from pytorch3d.io import load_obj, load_ply
from utils.file_management import get_unique_file
from data.amass_dataset import AmassDataset

def color_map(value, multiplier=1):
    value = value.cpu() * multiplier
    rgb = plt.cm.jet(value)[:,:-1]
    return rgb

# adapted from pyrender.Viewer
def _compute_camera_pose(centroid=0, scale=2.0):

    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3,:3] = np.array([
        [0.0, -s2, s2],
        [1.0, 0.0, 0.0],
        [0.0, s2, s2]
    ])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3,3] = dist * np.array([1.0, 0.0, 1.0]) + centroid

    return cp

def _renderSeq(meshes: list[trimesh.Trimesh], folder):
    renderer = pyrender.OffscreenRenderer(512, 512)
    with Pool(8) as p:
        meshes = p.map(pyrender.Mesh.from_trimesh, meshes)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1])
    x=-np.pi/4

    cam_pose = _compute_camera_pose()

    for i, mesh in enumerate(meshes):
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
        scene.add(mesh, pose=np.eye(4))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=np.eye(4))
        arr, _ = renderer.render(scene, pyrender.RenderFlags.SKIP_CULL_FACES)
        img = Image.fromarray(arr)
        img.save(os.path.join(folder, f"img{i}.png"))

    renderer.delete()

def export_video(tmp_dir, outputfile="./out.mp4", fps=50):
    os.system(f"ffmpeg -r {fps} -i {tmp_dir}/img%01d.png -pix_fmt yuv420p -b 5000k -vcodec h264 -y {outputfile}")

def save_video(sequence: list[trimesh.Trimesh], outputfile="./out.mp4", fps=50):
    with tempfile.TemporaryDirectory() as tmp_dir:
        _renderSeq(sequence, tmp_dir)
        export_video(tmp_dir, outputfile, fps)

def show_sequence(vertices, faces, frame_interval=40):
    scene = trimesh.Scene()
    for i in list(range(0, vertices.shape[0]-1, frame_interval)) + [vertices.shape[0]-1]:
        mesh = trimesh.Trimesh(vertices=vertices[i, :, :], faces=faces, process=False)
        scene.add_geometry(mesh)
    view = scene.show()
    display(view)
    return view

def render(mesh:trimesh.Trimesh, viewer='gl', record=False):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi*.5, [0,0,1]))

    if viewer or record and viewer not in ['gl', 'notebook']:
        Record(mesh, run_in_thread=False, from_start=record)
    else:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi*.5, [1,0,0]))
        display(mesh.show(viewer=viewer, smooth=False, kwargs={'cull': False}))

def imshow(img, offset=0, multiplier=1, rot=True):
    img = img * multiplier
    img[img!=0] += offset
    img = img.clamp(0, 1) * 255
    img = img.detach().cpu().numpy().astype(np.uint8)
    if rot:
        img = np.rot90(img)
    display(Image.fromarray(img))

def moving_average(values, window_size):
    if window_size <= 1:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

def plot_dict(dict, title="", xlabel="Step", ylabel="Value", legend=True, normalize=True, smoothing_window=1, save=False, dirname='./plots/'):

    plt.figure(figsize=(8, 5))
    for label, values in dict.items():
        values = list(values)
        values = moving_average(values, smoothing_window)

        # Normalize values if required
        if normalize:
            max_val = max(values)
            min_val = min(values)
            diff = max_val - min_val
            norm_values = [v - min_val for v in values]
            if diff == 0:
                norm_values = values  # Avoid divide by zero
            else:
                norm_values = [v / diff for v in values]
        else:
            norm_values = values
            max_val = None
        
        # Plot
        plt.plot(norm_values, label=f"{label} (min={min_val:.2f}, max={max_val:.2f})" if normalize else label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend:
        plt.legend(loc='best')

    if save:
        os.makedirs(dirname, exist_ok=True)
        file = get_unique_file(os.path.join(dirname, title), '.png')
        plt.savefig(file, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

class Record(pyrender.Viewer):
    def __init__(self, mesh=trimesh.Trimesh([[0,0,0]],[[0,0,0]]), from_start=False, output_video="./out.mp4", viewport_size=(1200, 800), run_in_thread=True, static_meshes=None):
        self.current_mesh = mesh
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene()
        scene.add(mesh, 'dynamic_meshes')
        if static_meshes is not None:
            scene.add(pyrender.Mesh.from_trimesh(static_meshes), 'static_meshes')

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.frame = 0
        self.output_video = output_video

        super().__init__(scene, use_raymond_lighting=True,
                        viewport_size=viewport_size,
                        cull_faces=False,
                        run_in_thread=run_in_thread)
        
        self.viewer_flags['record'] = from_start

    def _record(self):
        """Save another frame for the video.
        """
        data = self._renderer.read_color_buf()
        img = Image.fromarray(data)
        img.save(os.path.join(self.tmp_dir.name, f"img{self.frame}.png"))
        self.frame+=1

    def save_gif(self, filename=None):
        """Save the stored GIF frames to a file.

        To use this asynchronously, run the viewer with the ``record``
        flag and the ``run_in_thread`` flags set.
        Kill the viewer after your desired time with
        :meth:`.Viewer.close_external`, and then call :meth:`.Viewer.save_gif`.

        Parameters
        ----------
        filename : str
            The file to save the GIF to. If not specified,
            a file dialog will be opened to ask the user where
            to save the GIF file.
        """

        export_video(self.tmp_dir.name, self.output_video, self.viewer_flags['refresh_rate'])
        self.frame = 0
        self._saved_frames = []
        self.tmp_dir.cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()

    def on_close(self):
        if self.frame > 0:
            self.save_gif()
        self.tmp_dir.cleanup()
        super().on_close()

    def update_mesh(self, mesh:trimesh.Trimesh):
        self.current_mesh = mesh
        self.render_lock.acquire()
        mesh = pyrender.Mesh.from_trimesh(mesh)

        for node in self.scene.mesh_nodes:
            if node.name is not None and 'dynamic_meshes' in node.name:
                node.mesh = mesh
                break
        
        self.render_lock.release()

class InteractiveVisualizer(Record):

    def __init__(self, init_mesh=None, seq_count=0, seq_len=0, from_start=False, output_video="./out.mp4", viewport_size=(1200, 800), run_in_thread=True):
        
        self.seq_idx = 0
        self.frame_idx = 0
        self.play = False
        self.seq_count = seq_count
        self.seq_len = seq_len

        super().__init__(init_mesh, from_start, output_video, viewport_size, run_in_thread)
        self.registered_keys[pyglet.window.key.UP] = self.next_motion
        self.registered_keys[pyglet.window.key.DOWN] = self.prev_motion
        self.registered_keys[pyglet.window.key.RIGHT] = self.next_frame
        self.registered_keys[pyglet.window.key.LEFT] = self.prev_frame
        self.registered_keys[pyglet.window.key.SPACE] = self.toggle_play
        self.viewer_flags['record'] = from_start

    def toggle_play(self, i):
        self.play = not self.play

    def on_draw(self):
        super().on_draw()
        if self.play:
            self.next_frame(0)

    def next_motion(self, i):
        self.frame_idx = 0
        self.seq_idx = self.seq_idx + 1
        if self.seq_idx >= self.seq_count:
            self.seq_idx = 0
        self.update_seq()

    def prev_motion(self, i):
        self.frame_idx = 0
        self.seq_idx = self.seq_idx - 1

        if self.seq_idx < 0:
            self.seq_idx = self.seq_count-1
        self.update_seq()

    def next_frame(self, i):
        self.frame_idx = self.frame_idx + 1
        if self.frame_idx >= self.seq_len:
            self.frame_idx = 0
        self.update_frame()

    def prev_frame(self, i):
        self.frame_idx = self.frame_idx - 1

        if self.frame_idx < 0:
            self.frame_idx = self.seq_len-1
        self.update_frame()

    def update_frame(self):
        pass

    def update_seq(self):
        pass

class DGarmentDatasetvisualizer(InteractiveVisualizer):

    def __init__(self, config, from_start=False, output_video="./out.mp4", viewport_size=(1200, 800), run_in_thread=True):
        self.dataset = AmassDataset(config, config["dataset"]["folder"])
        self.faces = self.dataset.body_model.get_faces().detach().cpu()
        self.cloth_faces = load_obj(config['template_path'], load_textures=False)[1].verts_idx
        self.seq = self.dataset[0]
        self.mat_idx = 0
        self.sequence = self.dataset.body_model(self.seq.data).detach().cpu()
        mesh = trimesh.Trimesh(self.sequence[0], self.faces)
        cloth_path = os.path.dirname(self.seq.npz_file)
        cloth_path = os.path.join(cloth_path, "simulation_0", "data0", "output", "out_%.06d.ply"%200)
        verts, _ = load_ply(cloth_path)
        self.cloth_color = verts.clone() * 0
        self.cloth_color[:,0] = 1
        mesh += trimesh.Trimesh(verts, self.cloth_faces, vertex_colors=self.cloth_color)

        super().__init__(mesh, len(self.dataset), len(self.sequence), from_start, output_video, viewport_size, run_in_thread)
        self.registered_keys[pyglet.window.key.NUM_ADD] = self.next_mat
        self.registered_keys[pyglet.window.key.NUM_SUBTRACT] = self.prev_mat

    def update_frame(self):
        mesh = trimesh.Trimesh(self.sequence[self.frame_idx], self.faces)
        
        cloth_path = os.path.dirname(self.seq.npz_file)
        cloth_path = os.path.join(cloth_path, f"simulation_{self.mat_idx}", "data0", "output", "out_%.06d.ply"%(200+self.frame_idx))
        verts, _ = load_ply(cloth_path)
        mesh += trimesh.Trimesh(verts, self.cloth_faces, vertex_colors=self.cloth_color)
        self.set_caption(f"{self.frame_idx}/{self.seq_len-1} {self.seq_idx}/{self.seq_count-1} {self.mat_idx+1}/3 {self.seq.npz_file}")

        self.update_mesh(mesh)

    def update_seq(self):
        self.seq = self.dataset[self.seq_idx]
        self.sequence = self.dataset.body_model(self.seq.data).detach().cpu()
        self.seq_len = len(self.sequence)
        self.update_frame()

    def next_mat(self, i):
        self.mat_idx = self.mat_idx + 1
        if self.mat_idx >= 3:
            self.mat_idx = 0
        self.update_seq()

    def prev_mat(self, i):
        self.mat_idx = self.mat_idx - 1

        if self.mat_idx < 0:
            self.mat_idx = 2
        self.update_seq()

class Sequencevisualizer(InteractiveVisualizer):

    def __init__(self, sequence, from_start=False, output_video="./out.mp4", viewport_size=(1200, 800), run_in_thread=True):
        self.sequence = sequence
        super().__init__(sequence[0], 1, len(sequence), from_start, output_video, viewport_size, run_in_thread)

    def update_frame(self):
        mesh = self.sequence[self.frame_idx]
        self.set_caption(str(self.frame_idx))
        self.update_mesh(mesh)

    def update_seq(self):
        self.update_frame()
