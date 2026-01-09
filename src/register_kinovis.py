import os
import json
import argparse
import torch
from tqdm.auto import tqdm
import trimesh
from torch.utils.data import DataLoader
import pickle
from body_models.smpl_model import SMPLH
from model.VDM_latent import VDMStableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from data.fitting_sequence import Fitting_Sequence
from data.structures import Mesh, GarmentDict, dataclass_collate
from losses.losses import collision_loss, clothing_energy, temporal_loss
from utils.geometry import batch_compute_points_normals
from utils.uv_tools import apply_displacement
from losses.objectives import chamfer_distance, py3d_point_mesh_face_distance
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
from pytorch3d.ops.knn import knn_points
from utils.visualization import Record, plot_dict
from utils.file_management import get_unique_folder
from enum import Flag, auto
from data.normalization import unnormalize_cloth, denormalize_material
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
import torch.nn.functional as F

class Loss(Flag):
    S2T_CHAMFER = auto()    # unidirectional chamfer distance from source to target
    T2S_CHAMFER = auto()    # unidirectional chamfer distance from target to source
    M2P = auto()            # mesh to point distance
    P2M = auto()            # point to mesh distance
    CLOTH_BODY_PEN = auto() # minimize cloth in body penetration
    L2_REG = auto()         # L2 regularization on latent
    SMOOTH = auto()         # laplacian smoothing regularization on surface
    EDGE_REG = auto()       # edge length regularization
    NORMAL = auto()         # normal similarity loss
    TEMPORAL = auto()       # temporal distance loss

def reconstruct(
    config,
    pipeline:VDMStableDiffusionPipeline,
    body_model:SMPLH,
    sequence:Fitting_Sequence,
    template:Mesh,
    unique_latent=False,
    concat_body=False,
    optimize_pose=False,
    optimize_trans=False,
    optimize_material=False,
    normalize_material=True,
    num_opti_epoch=50,
    diffusion_steps=10,
    batch_size=1,
    lr=1e-3,
    t2s_weight=0.,
    s2t_weight=0.,
    p2m_weight=0.,
    m2p_weight=0.,
    reg_weight=0.,
    normal_weight=0.,
    pen_weight=0.,
    smooth_weight=0.,
    edge_weight=0.,
    temporal_weight=0.,
    # t2s_weight=1.,
    # s2t_weight=0.1,
    # reg_weight=1e2,
    # normal_weight=1e-3,
    # pen_weight=1e0,
    # smooth_weight=1e0,
    visualize=False,
    seed=0,
    rolls=0,
    latent_file=None
):
    device = pipeline.device
    dtype = pipeline.dtype
    batch_size = min(batch_size, len(sequence))
    view_indices = torch.tensor([0, len(sequence)//2, len(sequence)-1]).unique()

    print(f"Using batch size {batch_size}")

    sequence_dataloader = DataLoader(
        sequence,
        batch_size=batch_size,
        collate_fn=dataclass_collate,
        shuffle=False,
        drop_last=False,)

    # when not normalizing this would be the mean of the training data
    # bending = torch.tensor(0.).to(device)
    # stretching = torch.tensor(0.).to(device)
    # density = torch.tensor(0.).to(device)

    # used values which give good results
    bending = torch.tensor(1e-8).to(device)
    stretching = torch.tensor(200.).to(device)
    density = torch.tensor(0.01).to(device)

    # B16_shape0_sim2 parameters after optimization
    # bending = torch.tensor(5.008578136909386e-05).to(device)
    # stretching = torch.tensor(163.96781728139698).to(device)
    # density = torch.tensor(0.05036955246331135).to(device)

    # C19_shape_1_sim1 parameters after optimization
    # target_bending = 1.1806429695163405e-05
    # target_stretching = 97.88990368938657
    # target_density = 0.5539877977551471

    # prepare variables
    optimized_parameters = []

    active_losses = Loss(0)

    if s2t_weight != 0.:
        active_losses |= Loss.S2T_CHAMFER
    if t2s_weight != 0.:
        active_losses |= Loss.T2S_CHAMFER
    if p2m_weight != 0.:
        active_losses |= Loss.P2M
    if m2p_weight != 0.:
        active_losses |= Loss.M2P
    if pen_weight != 0.:
        active_losses |= Loss.CLOTH_BODY_PEN
    if reg_weight != 0.:
        active_losses |= Loss.L2_REG
    if smooth_weight != 0.:
        active_losses |= Loss.SMOOTH
    if normal_weight != 0.:
        active_losses |= Loss.NORMAL
    if edge_weight != 0.:
        active_losses |= Loss.EDGE_REG
    if temporal_weight != 0.:
        active_losses |= Loss.TEMPORAL

    register_losses = dict()
    for k in active_losses:
        register_losses[k.name] = []
    # register_losses['robust_alpha'] = []
    # register_losses['robust_scale'] = []

    if latent_file is not None:
        optimized_latents = torch.load(latent_file).to(device)
    else:
        min_chamfer = 1e12
        chmlist = []
        with torch.no_grad():
            for roll in tqdm(range(seed, seed+rolls), desc="Rolling seeds"):
                test_latents = pipeline.prepare_latents(1, torch.Generator(device='cpu').manual_seed(roll))

                chamfer = 0
                for step, batch in enumerate(sequence_dataloader):
                    batch = batch.to(device, torch.float32)
                    batch_size = batch.poses.shape[0]

                    conditions = GarmentDict(
                        betas = batch.betas,
                        poses = batch.poses,
                        trans = batch.trans,
                        root_joint_position = batch.root_joint_position,
                        bending = bending[None, ...].expand(batch_size, -1),
                        stretching = stretching[None, ...].expand(batch_size, -1),
                        density = density[None, ...].expand(batch_size, -1)
                    )

                    latents_batched = test_latents.expand(batch_size, -1, -1, -1)
                    image = pipeline(conditions,
                                    num_inference_steps=diffusion_steps,
                                    latents=latents_batched,
                                    gradient=False,
                                    normalize_material=normalize_material
                                    )

                    # compute loss
                    vertices = apply_displacement(template, image)
                    conditions.cloth_vertices = vertices
                    vertices = unnormalize_cloth(conditions)

                    _, _, _, _, dist_x, dist_y = chamfer_distance(batch.sampled_vertices, vertices)
                    chamfer += dist_x.sum() + dist_y.sum()
                chmlist.append(chamfer.item())
                if chamfer < min_chamfer:
                    seed = roll
                    min_chamfer = chamfer
        print(f"Using seed {seed}")

        optimized_latents = pipeline.prepare_latents(1, torch.Generator(device='cpu').manual_seed(seed))

    init_latent = optimized_latents.detach().clone()
    
    if not unique_latent:
        optimized_latents = optimized_latents.expand(len(sequence), -1, -1, -1).clone()

    optimized_latents = torch.nn.Parameter(optimized_latents)
    optimized_parameters.append(optimized_latents)

    if optimize_material:
        normalize_material = False
        bending = torch.nn.Parameter(torch.tensor(0.).to(device))
        stretching = torch.nn.Parameter(torch.tensor(0.).to(device))
        density = torch.nn.Parameter(torch.tensor(0.).to(device))
        optimized_parameters.extend([bending, stretching, density])

        mat_tuple = denormalize_material(bending, stretching, density)
        b_val, s_val, d_val = tuple(x.item() for x in mat_tuple)
        register_losses['b'] = []
        register_losses['s'] = []
        register_losses['d'] = []
        print (f"bending: {b_val}, stretching: {s_val}, density: {d_val}")

    if optimize_pose:
        sequence.data['poses'] = torch.nn.Parameter(sequence.data['poses'])
        optimized_parameters.append(sequence.data['poses'])
        register_losses['body_clothing'] = []
        sequence.update_smpl_meshes() # update grads if needed
    if optimize_trans:
        sequence.data['trans'] = torch.nn.Parameter(sequence.data['trans'])
        optimized_parameters.append(sequence.data['trans'])
        sequence.update_smpl_meshes() # update grads if needed

    if visualize:
        meshes = []
        for m in sequence.meshes[view_indices]:
            mesh = trimesh.Trimesh(m.verts_packed().cpu(), m.faces_packed().cpu())
            meshes.append(mesh)

        mesh = trimesh.util.concatenate(meshes)
        viewer = Record(static_meshes=mesh)
        body_faces = body_model.get_faces().detach().cpu()

    robust_funcs = AdaptiveLossFunction(1, torch.float32, device, alpha_init=0.001, alpha_lo=1e-7, scale_lo=1e-3)
    optimized_parameters.extend(list(robust_funcs.parameters()))

    optimizer = torch.optim.Adam(optimized_parameters, lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_opti_epoch)
    progress_bar = tqdm(total=num_opti_epoch*len(sequence_dataloader))

    # optimization loop
    for epoch in range(num_opti_epoch):
        for step, batch in enumerate(sequence_dataloader):
            bs = batch.poses.shape[0]

            conditions = GarmentDict(
                betas = batch.betas,
                poses = batch.poses,
                trans = batch.trans,
                root_joint_position = batch.root_joint_position,
                bending = bending[None, ...].expand(bs, -1),
                stretching = stretching[None, ...].expand(bs, -1),
                density = density[None, ...].expand(bs, -1)
            )
            if unique_latent:
                latents_batched = optimized_latents.expand(bs, -1, -1, -1)
            else:
                latents_batched = optimized_latents[batch.index]

            image = pipeline(conditions,
                            num_inference_steps=diffusion_steps,
                            latents=latents_batched,
                            gradient=True,
                            gradient_unet=optimize_material,
                            normalize_material=normalize_material
                            )

            # deform template
            vertices = apply_displacement(template, image)
            conditions.cloth_vertices = vertices
            cloth_vertices = unnormalize_cloth(conditions)
            cloth_normals = batch_compute_points_normals(cloth_vertices, template.f)
            cloth_meshes = Meshes(cloth_vertices, template.f[None, ...].expand(bs, -1, -1))

            # compute loss
            loss = torch.tensor(0., device=device)

            # concat body + cloth points
            vertices = cloth_vertices
            if concat_body:
                vertices = torch.cat((cloth_vertices, batch.body_vertices), dim=-2)

            if Loss.T2S_CHAMFER in active_losses:
                knn = knn_points(batch.sampled_vertices, vertices)
                dists = knn.dists
                dists = robust_funcs(dists.flatten()[:, None])
                dists = dists.mean() * t2s_weight
                loss += dists
                register_losses[Loss.T2S_CHAMFER.name].append(dists.item())

            if Loss.S2T_CHAMFER in active_losses:
                knn = knn_points(vertices, batch.sampled_vertices)
                dists = knn.dists
                dists = robust_funcs(dists.flatten()[:, None])
                dists = dists.mean() * s2t_weight
                loss += dists
                register_losses[Loss.S2T_CHAMFER.name].append(dists.item())

            if Loss.M2P in active_losses:
                point_to_face, face_to_point = py3d_point_mesh_face_distance(cloth_meshes, Pointclouds(batch.sampled_vertices), True)
                dist = robust_funcs(face_to_point.flatten()[:, None])
                dist = dist.mean() * m2p_weight
                loss += dist
                register_losses[Loss.M2P.name].append(dist.item())

            if Loss.P2M in active_losses:
                if not Loss.M2P in active_losses:
                    point_to_face, face_to_point = py3d_point_mesh_face_distance(cloth_meshes, Pointclouds(batch.sampled_vertices), False)
                dist = robust_funcs(point_to_face.flatten()[:, None])
                dist = dist.mean() * p2m_weight
                loss += dist
                register_losses[Loss.P2M.name].append(dist.item())

            if Loss.CLOTH_BODY_PEN in active_losses:
                # sampling at each step so we actually optimize face colision instead of disconnected points
                sampled_cloth_vertices, cloth_normals = sample_points_from_meshes(cloth_meshes, num_samples=5000, return_normals=True)
                penetration = collision_loss(sampled_cloth_vertices, batch.body_vertices, batch.body_normals, cloth_normals).mean() * pen_weight
                loss += penetration
                register_losses[Loss.CLOTH_BODY_PEN.name].append(penetration.item())

            if Loss.SMOOTH in active_losses:
                reg_loss = mesh_laplacian_smoothing(cloth_meshes, 'uniform') * smooth_weight # cotcurv, uniform, cot
                loss += reg_loss
                register_losses[Loss.SMOOTH.name].append(reg_loss.item())

            if Loss.L2_REG in active_losses:
                reg_loss = F.mse_loss(init_latent.expand(bs, -1, -1, -1), latents_batched)
                reg_loss = reg_loss.mean() * reg_weight
                loss += reg_loss
                register_losses[Loss.L2_REG.name].append(reg_loss.item())

            if Loss.TEMPORAL in active_losses:
                verts_padded = cloth_meshes.verts_padded()
                if step != 0:
                    verts_padded = torch.cat((last_temp, verts_padded))
                reg_loss = temporal_loss(verts_padded)
                reg_loss = reg_loss.mean() * temporal_weight
                loss += reg_loss
                register_losses[Loss.TEMPORAL.name].append(reg_loss.item())
                last_temp = verts_padded[-1][None,...].detach()

            if Loss.EDGE_REG in active_losses:
                reg_loss = mesh_edge_loss(cloth_meshes) * edge_weight
                loss += reg_loss
                register_losses[Loss.EDGE_REG.name].append(reg_loss.item())

            if Loss.NORMAL in active_losses:
                normal_loss = mesh_normal_consistency(cloth_meshes) * normal_weight
                loss += normal_loss
                register_losses[Loss.NORMAL.name].append(normal_loss.item())

            if optimize_pose:
                body_clothing = clothing_energy(batch.body_vertices, batch.sampled_vertices, batch.sampled_normals)
                body_clothing = body_clothing.mean()
                loss += body_clothing
                register_losses['body_clothing'].append(body_clothing.item())

            loss = loss.to(dtype=torch.float32)
            loss.backward(retain_graph=False)

            # CHECK FOR NANS
            # torch.nn.utils.clip_grad_norm_(latents, 1.0)
            # print(torch.isnan(latents).any())
            # for name, param in self.unet.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"unet NaNs in gradient of {name}")
            # for name, param in self.vae.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"vae NaNs in gradient of {name}")

            # register_losses['robust_alpha'].append(robust_func.alpha()[0,0].item())
            # register_losses['robust_scale'].append(robust_func.scale()[0,0].item())
            
            if optimize_material:
                b_val, s_val, d_val = denormalize_material(bending.detach(), stretching.detach(), density.detach())
                register_losses['b'].append(b_val.item())
                register_losses['s'].append(s_val.item())
                register_losses['d'].append(d_val.item())

            postfix = dict()
            for k in register_losses.keys():
                postfix[k] = register_losses[k][-1]
            progress_bar.set_postfix(postfix)

            if visualize and epoch%5==0:
                common_idx = view_indices[torch.isin(view_indices, batch.index)]

                if common_idx.numel() > 0:
                    cpu_vertices = cloth_vertices[common_idx%batch_size].detach().cpu()
                    faces = template.f.detach().cpu()
                    rgb = torch.zeros_like(cpu_vertices[0])
                    rgb[..., 1] = 1
                    brgb = torch.zeros_like(sequence.body_vertices[0].detach(), device='cpu')
                    brgb[..., 0] = 1
                    meshes = []
                    if step!=0:
                        meshes.append(viewer.current_mesh)
                    for v, bv in zip(cpu_vertices, sequence.body_vertices[common_idx].detach().cpu()):
                        mesh = trimesh.Trimesh(v, faces, vertex_colors=rgb)
                        mesh += trimesh.Trimesh(bv, body_faces, vertex_colors=brgb)
                        meshes.append(mesh)

                    viewer.update_mesh(trimesh.util.concatenate(meshes))

            if optimize_pose or optimize_trans:
                sequence.update_smpl_meshes() # update and clean grads

            progress_bar.update()

        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()

    progress_bar.close()

    chamfer_list = []
    with torch.no_grad():
        sequence_cloth_vertices = []
        for step, batch in enumerate(sequence_dataloader):
            batch_size = batch.poses.shape[0]

            conditions = GarmentDict(
                betas = batch.betas,
                poses = batch.poses,
                trans = batch.trans,
                root_joint_position = batch.root_joint_position,
                bending = bending[None, ...].expand(batch_size, -1),
                stretching = stretching[None, ...].expand(batch_size, -1),
                density = density[None, ...].expand(batch_size, -1)
            )

            if unique_latent:
                latents_batched = optimized_latents.expand(batch_size, -1, -1, -1)
            else:
                latents_batched = optimized_latents[batch.index]

            image = pipeline(conditions,
                            num_inference_steps=diffusion_steps,
                            latents=latents_batched,
                            gradient=False,
                            normalize_material=normalize_material
                            )

            # compute loss
            vertices = apply_displacement(template, image)
            conditions.cloth_vertices = vertices
            vertices = unnormalize_cloth(conditions)
            sequence_cloth_vertices.append(vertices)
            _, _, _, _, dist_x, dist_y = chamfer_distance(batch.sampled_vertices, vertices)
            chamfer = (dist_x.mean(dim=1)+dist_y.mean(dim=1))*0.5
            chamfer_list.append(chamfer)

    print("Mean Chamfer: " + str(chamfer_list))

    chamfer_list = torch.cat(chamfer_list, dim=0)

    sequence_cloth_vertices = torch.cat(sequence_cloth_vertices).detach()

    if optimize_material:
        mat_tuple = denormalize_material(bending, stretching, density)
        b_val, s_val, d_val = tuple(x.item() for x in mat_tuple)
        
        print (f"bending: {b_val}, stretching: {s_val}, density: {d_val}")

    return sequence_cloth_vertices, optimized_latents.detach(), chamfer_list.detach(), sequence, register_losses


def compute_sequence(config:str,
                    sequence_path:str,
                    out_folder:str,
                    batch_size=1,
                    lr=1e-4,
                    t2s_weight=0.,
                    s2t_weight=0.,
                    p2m_weight=0.,
                    m2p_weight=0.,
                    reg_weight=0.,
                    normal_weight=0.,
                    pen_weight=0.,
                    smooth_weight=0.,
                    temporal_weight=0.,
                    edge_weight=0.,
                    num_opti_epoch=50,
                    diffusion_steps=10,
                    num_samples=1000,
                    subdivide=False,
                    visualize=False,
                    unique_latent=False,
                    concat_body=False,
                    filter_capture=False,
                    optimize_material=False,
                    optimize_pose=False,
                    optimize_trans=False,
                    start_idx=0,
                    end_idx=None,
                    seed=0,
                    rolls=0,
                    latent_file=None,
                    args=""):
    '''
    fit D-Garment to captured clothed human
    cong: str path to configuration file
    sequence_path: str path to the folder containing a sequence of `ply` files and `pose.npz` precomputed SMPL fitting
    cong: str output folder where the fitted cloth and latent will be saved
    '''

    with open(config) as f:
        config = json.load(f)

    device = config['device']
    dtype = torch.bfloat16 if config['mixed_precision']=='bf16' else torch.float16 if config['mixed_precision']=='fp16' else torch.float32

    template = Mesh(config, device=device, subdivide=subdivide)
    body_model = SMPLH(config).to(device)
    captured_sequence = Fitting_Sequence(sequence_path, body_model=body_model, config=config, device=device, start=start_idx, end=end_idx, num_samples=num_samples)

    if filter_capture:
        captured_sequence.remove_capture_outliers(config['body_model']['smpl_seg_file'])

    pipeline = VDMStableDiffusionPipeline.from_pretrained(config["output"]["folder"], torch_dtype=dtype).to(device)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="dpmsolver++")
    # pipeline.compile()
    # pipeline.enable_vae_slicing()

    pipeline.set_progress_bar_config(disable=True)
    reconstructed,latent, chamfer, sequence, register_losses = reconstruct(
        config,
        pipeline,
        body_model,
        captured_sequence,
        template,
        batch_size=batch_size,
        lr=lr,
        t2s_weight=t2s_weight,
        s2t_weight=s2t_weight,
        p2m_weight=p2m_weight,
        m2p_weight=m2p_weight,
        reg_weight=reg_weight,
        normal_weight=normal_weight,
        temporal_weight=temporal_weight,
        pen_weight=pen_weight,
        smooth_weight=smooth_weight,
        edge_weight=edge_weight,
        num_opti_epoch=num_opti_epoch,
        diffusion_steps=diffusion_steps,
        unique_latent=unique_latent,
        concat_body=concat_body,
        optimize_material=optimize_material,
        optimize_pose=optimize_pose,
        optimize_trans=optimize_trans,
        visualize=visualize,
        seed=seed,
        rolls=rolls,
        latent_file=latent_file
    )

    # save outputs
    out_folder = get_unique_folder(out_folder)

    os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(out_folder, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=1)

    if num_opti_epoch > 0:
        with open(os.path.join(out_folder, 'losses.pkl'), 'wb') as f:
            pickle.dump(register_losses, f)
        plot_dict(register_losses, "Fitting", save=True, dirname=out_folder, smoothing_window=10)
    
    torch.save(latent.detach().cpu(), os.path.join(out_folder, "latent.pt"))
    torch.save(chamfer.detach().cpu(), os.path.join(out_folder, "chamfer.pt"))

    sequence.save_smpl(os.path.join(out_folder, "poses.npz"))

    out_body_folder = os.path.join(out_folder, "body_meshes")
    os.makedirs(out_body_folder, exist_ok=True)
    out_cloth_folder = os.path.join(out_folder, "cloth_meshes")
    os.makedirs(out_cloth_folder, exist_ok=True)

    faces = template.f.cpu()
    body_faces = body_model.get_faces().detach().cpu()
    body_vertices = body_model(sequence.smpl_data()).detach().cpu()

    for i, v in enumerate(reconstructed.cpu()):
        body = trimesh.Trimesh(body_vertices[i], body_faces, process=False)
        body.export(os.path.join(out_body_folder, f"{i}.ply"))

        cloth = trimesh.Trimesh(v, faces, process=False)
        cloth.export(os.path.join(out_cloth_folder, f"{i}.ply"))
        

'''
Script to fit the model to simulated cloth sequence
'''
if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    '''
    from pytorch3d tutorial
    torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 
    '''
    description = 'Script to fix cloth intersecting SMPL in dataset'
    parser = argparse.ArgumentParser(description=description,
                                     prog='Batch-fix')
    parser.add_argument('config', type=str,
                        help='The config file')
    parser.add_argument('sequence_path', type=str,
                        help='The folder containing smpl poses.npz and a sequence of captured meshes')
    parser.add_argument('--output', default='./out', type=str,
                        help='The output folder where the fitted cloth and latent will be saved')
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='Learning rate')
    parser.add_argument('--t2s_weight', default=1., type=float,
                        help='Target to surface weight')
    parser.add_argument('--s2t_weight', default=0.5, type=float,
                        help='Surface to target weight')
    parser.add_argument('--p2m_weight', default=0., type=float,
                        help='Point to mesh weight')
    parser.add_argument('--m2p_weight', default=0., type=float,
                        help='Mesh to point weight')
    parser.add_argument('--normal_weight', default=0.001, type=float,
                        help='Normal t2s weight')
    parser.add_argument('--pen_weight', default=1., type=float,
                        help='Cloth-body penetration weight')
    parser.add_argument('--reg_weight', default=0., type=float,
                        help='L2 latent regularization weight')
    parser.add_argument('--smooth_weight', default=0.001, type=float,
                        help='Laplacian smoothing weight')
    parser.add_argument('--edge_weight', default=.1, type=float,
                        help='Cloth edge length regularization weight')
    parser.add_argument('--temporal_weight', default=0., type=float,
                        help='Temporal smoothing regularization weight')
    parser.add_argument('--start', default=0, type=int,
                        help='Only optimize the from given frame index')
    parser.add_argument('--end', default=None, type=int,
                        help='Only optimize to the given frame index')
    parser.add_argument('--num_samples', default=10000, type=int,
                        help='Sampled points')
    parser.add_argument('--diffusion_steps', default=20, type=int,
                        help='Number of diffusion steps')
    parser.add_argument('--epochs', default=500, type=int,
                        help='Number of optimization epochs')
    parser.add_argument("--subdivide", action=argparse.BooleanOptionalAction,
                        help="subdivide")
    parser.add_argument("--unique_latent", action=argparse.BooleanOptionalAction,
                        help="Use a single latent for the whole sequence")
    parser.add_argument("--concat_body", action=argparse.BooleanOptionalAction,
                        help="Wheras the Chamfer is computed with body vertices")
    parser.add_argument("--filter_capture", action=argparse.BooleanOptionalAction,
                        help="Wheras capture outliers are removed using smpl segmentation")
    parser.add_argument("--optimize_material", action=argparse.BooleanOptionalAction,
                        help="Optimize cloth material parameter")
    parser.add_argument("--optimize_pose", action=argparse.BooleanOptionalAction,
                        help="Optimize SMPL pose parameter")
    parser.add_argument("--optimize_trans", action=argparse.BooleanOptionalAction,
                        help="Optimize SMPL trans parameter")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction,
                        help="visualize")
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--rolls', default=0, type=int,
                        help='Number of random seeds to test to keep the best one')
    parser.add_argument('--latent', default=None, type=str,
                        help='Start from latent in .pt file')

    args = parser.parse_args()

    compute_sequence(config=args.config,
                    sequence_path=args.sequence_path,
                    out_folder=args.output,
                    batch_size=args.batch_size,
                    diffusion_steps=args.diffusion_steps,
                    lr=args.lr,
                    t2s_weight=args.t2s_weight,
                    s2t_weight=args.s2t_weight,
                    p2m_weight=args.p2m_weight,
                    m2p_weight=args.m2p_weight,
                    reg_weight=args.reg_weight,
                    normal_weight=args.normal_weight,
                    pen_weight=args.pen_weight,
                    smooth_weight=args.smooth_weight,
                    edge_weight=args.edge_weight,
                    temporal_weight=args.temporal_weight,
                    num_opti_epoch=args.epochs,
                    num_samples=args.num_samples,
                    subdivide=args.subdivide,
                    unique_latent=args.unique_latent,
                    concat_body=args.concat_body,
                    filter_capture=args.filter_capture,
                    optimize_material=args.optimize_material,
                    optimize_pose=args.optimize_pose,
                    optimize_trans=args.optimize_trans,
                    visualize=args.visualize,
                    seed=args.seed,
                    rolls=args.rolls,
                    latent_file=args.latent,
                    start_idx=args.start,
                    end_idx=args.end,
                    args=args)