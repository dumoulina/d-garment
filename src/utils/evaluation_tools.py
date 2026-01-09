import torch
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import  knn_points


def chamfer_l1(xpc, ypc):
    x, x_lengths, x_normals = _handle_pointcloud_input(
        xpc, None, None)
    y, y_lengths, y_normals = _handle_pointcloud_input(
        ypc, None, None)

    x_nn = knn_points(x, y, lengths1=x_lengths,
                      lengths2=y_lengths, K=1)

    y_nn = knn_points(y, x, lengths1=y_lengths,
                      lengths2=x_lengths, K=1)

    cham_y = x_nn.dists[..., 0]
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    return (torch.mean(torch.sqrt(cham_x[cham_x != 0]).squeeze())+torch.mean(torch.sqrt(cham_y[cham_y != 0]).squeeze()))*0.5


def compute_errors(ref_seq, test_seqs, angular=False, body_model=None, nsteps=100):

    if(not angular and body_model is None):
        raise Exception("If angular is False, specify a body model")

    errors = []
    duration_errors = []
    timesteps = torch.linspace(0, ref_seq.duration(), nsteps).tolist()

    frames_ref = ref_seq.frame_data(timesteps, unit="s")
    meshes_ref = body_model(frames_ref)

    for seq in test_seqs:
        frames = seq.frame_data(timesteps, unit="s")
        meshes = body_model(frames)

        if(angular):
            errors.append(torch.sqrt(torch.square(
                frames["poses"]-frames_ref["poses"])))

        else:
            errors.append(torch.sqrt(
                torch.sum(torch.pow((meshes-meshes_ref), 2), dim=-1)))

        duration_errors.append(abs(seq.duration()-ref_seq.duration()))

    return errors, duration_errors


def compute_errors_kino(kino_seq, amass_seqs, body_model, nsteps=100):

    errors = []
    duration_errors = []
    timesteps = torch.linspace(0, kino_seq.duration(), nsteps).tolist()

    frames_ref = kino_seq.frame_data(timesteps, unit="s", aligned=True)
    meshes_kino = [torch.tensor(m.vertices, dtype=torch.float32, device=amass_seqs[0].device)
                   for m in frames_ref["meshes"]]

    for seq in amass_seqs:
        frames = seq.frame_data(timesteps, unit="s")
        meshes_amass = body_model(frames)

        errors.append(torch.zeros(len(meshes_amass), 6890))
        for i, (mk, ma) in enumerate(zip(meshes_kino, meshes_amass)):

            x, x_lengths, x_normals = _handle_pointcloud_input(
                ma[None, :, :], None, None)
            y, y_lengths, y_normals = _handle_pointcloud_input(
                mk[None, :, :], None, None)

            N, P1, D = x.shape
            P2 = y.shape[1]

            cham_norm_x = x.new_zeros(())
            cham_norm_y = x.new_zeros(())

            x_nn = knn_points(x, y, lengths1=x_lengths,
                              lengths2=y_lengths, K=1)
            y_nn = knn_points(y, x, lengths1=y_lengths,
                              lengths2=x_lengths, K=1)

            cham_x = x_nn.dists[..., 0]  # (N, P1)
            cham_y = y_nn.dists[..., 0]  # (N, P2)

            errors[-1][i] = cham_x.squeeze()

    duration_errors.append(abs(seq.duration()-kino_seq.duration()))

    return errors, duration_errors

