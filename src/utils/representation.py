import torch
import torchgeometry as tgm
import utils.utils_euler as t3d
AA = {"name": "aa", "dim": 3}
SIX = {"name": "six", "dim": 6}
MAT = {"name": "mat", "dim": 9}
QUAT = {"name": "quat", "dim": 4}
EUL = {"name": "euler", "dim": 3}

# BEGIN error def

class UnknownConversionException(Exception):
    pass

class InvalidShapeException(Exception):
    pass

def checkTensor(*args): 
    for i, arg in enumerate(args):
        if not torch.is_tensor(arg):
            raise TypeError("Expected Tensor, got {} (arg nb {})".format(type(arg),i))

def checkNumDims(*args):
    
    ref_num_dim = len(args[0].shape)

    for arg in args[1:]:
        if(len(arg.shape)!= ref_num_dim):
            raise InvalidShapeException("Tensors should have same number of dims")
        
# END error def

def extractAxisAngle(pose, eps=1e-6):
    """ extract intuitive axis angle components from aa representation
        pose : ... x 3
    """
    checkTensor(pose)

    if pose.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3) got {}".format(pose.shape))

    # amplitude of the rotation in radiants
    angle = pose.norm(dim=-1, p='fro', keepdim=True)

    # rotation around the axis (ax, ay, az)
    p = pose / (angle + eps)
    ax = p[..., 0:1]
    ay = p[..., 1:2]
    az = p[..., 2:]

    return angle.contiguous(), ax.contiguous(), ay.contiguous(), az.contiguous()


def rotMatCoord(angle, x, y, z):
    """rot mat components from parameters of intuitive axis angle representation"""
    checkTensor(angle, x, y, z)

    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c

    r11 = t*x*x + c
    r12 = t*x*y - z*s
    r13 = t*x*z + y*s
    r21 = t*x*y + z*s
    r22 = t*y*y + c
    r23 = t*y*z - x*s
    r31 = t*x*z - y*s
    r32 = t*y*z + x*s
    r33 = t*z*z + c

    return r11, r12, r13, r21, r22, r23, r31, r32, r33


def aa2quat(pose):
    """ Convert a pose tensor with axis angle representation ( ... x 3)
        into a quaternion representation (... x 4 )"""
    checkTensor(pose)
    if pose.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3) got {}".format(pose.shape))

    angle, ax, ay, az = extractAxisAngle(pose)

    s = torch.sin(angle*.5)
    qx = ax * s
    qy = ay * s
    qz = az * s
    qw = torch.cos(angle*.5)

    q = torch.cat((qw, qx, qy, qz), -1)

    return q.contiguous()


def quat2rotmat(pose):
    """ Convert a pose tensor with quaternion representation (..., 4)
        into a rotmat representation (..., 3 , 3 )"""

    checkTensor(pose)
    if pose.shape[-1:] != (4,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,4) got {}".format(pose.shape))

    qx = pose[..., 1:2]
    qy = pose[..., 2:3]
    qz = pose[..., 3:]
    qw = pose[..., 0:1]

    qz2 = torch.pow(qz, 2)
    qx2 = torch.pow(qx, 2)
    qy2 = torch.pow(qy, 2)

    r11 = 1 - 2*qy2 - 2 * qz2
    r12 = 2*qx*qy - 2*qz*qw
    r13 = 2*qx*qz + 2*qy*qw
    r21 = 2*qx*qy + 2*qz*qw
    r22 = 1 - 2*qx2 - 2*qz2
    r23 = 2*qy*qz - 2*qx*qw
    r31 = 2*qx*qz - 2*qy*qw
    r32 = 2*qy*qz + 2*qx*qw
    r33 = 1 - 2*qx2 - 2*qy2

    rotmat = torch.cat((r11, r12, r13,
                        r21, r22, r23,
                        r31, r32, r33), -1)

    shape = list(rotmat.shape)
    shape[-1] = 3
    shape.append(3)

    return rotmat.contiguous().view(shape)


def rotmat2quat(pose, eps=1e-6):
    """Convert rotation matrix to quaternion , correct a bug from torchgeometry

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Code from pygeometry library, had to reimplement the negation of the masks
    because there is an error in the pygeometry library

    pose : ... x 3 x 3
    """

    checkTensor(pose)
    if pose.shape[-2:] != (3, 3):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3,3) got {}".format(pose.shape))

    out_shape = list(pose.shape[:-1])
    out_shape[-1] = 4

    rmat_t = pose.contiguous().view(-1, 3, 3)
    rmat_t = torch.transpose(rmat_t, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5

    return q.contiguous().view(out_shape)


def quat2aa(pose):
    """ Convert a pose tensor with quaternion representation (... x 4)
        into an axis angle representation (... x 3)"""
    checkTensor(pose)
    if pose.shape[-1:] != (4,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,4) got {}".format(pose.shape))

    out_shape = list(pose.shape)
    out_shape[-1] = 3

    rp = pose.contiguous().view(-1, 4)

    return tgm.quaternion_to_angle_axis(rp).contiguous().view(out_shape)


def aa2rotmat(pose):
    """ Convert a pose tensor with axis-angle representation (..., 3)
        into a rotation matrix representation (..., 3 , 3 )"""
    checkTensor(pose)

    if pose.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3) got {}".format(pose.shape))

    return quat2rotmat(aa2quat(pose))


def rotmat2six(pose):
    """ Convert a pose tensor with rotation matrix representation (..., 3 , 3))
        into the 6D representation (..., 6)
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""

    out_shape = list(pose.shape[:-1])
    out_shape[-1] = 6

    # keeping only the first 2 columns
    a1 = pose[..., :, 0]
    a2 = pose[..., :, 1]

    return torch.cat((a1, a2), -1).view(out_shape)


def cross_product(u, v):
    """ Returns the cross product between two (..., 3) vectors """
    checkTensor(u, v)
    checkNumDims(u, v)
    if u.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid vector shape, expected (...,3) got {}".format(u.shape))
    if v.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid vector shape, expected (...,3) got {}".format(v.shape))

    i = u[..., 1:2]*v[..., 2:] - u[..., 2:]*v[..., 1:2]
    j = u[..., 2:]*v[..., 0:1] - u[..., 0:1]*v[..., 2:]
    k = u[..., 0:1]*v[..., 1:2] - u[..., 1:2]*v[..., 0:1]

    return torch.cat((i, j, k), -1)


def six2rotmat(pose, eps=1e-6):
    """ Convert a pose tensor with 6D representation (..., 6)
        into the rotation matrix representation (..., 9)
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""
    checkTensor(pose)
    if pose.shape[-1:] != (6,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,6) got {}".format(pose.shape))

    x_raw = pose[..., 0:3]
    y_raw = pose[..., 3:6]

    x = x_raw / (x_raw.norm(dim=-1, p='fro', keepdim=True) + eps)
    z = cross_product(x, y_raw)
    z = z / (z.norm(dim=-1, p='fro', keepdim=True) + eps)
    y = cross_product(z, x)

    return torch.stack((x, y, z), -1)


def rotmat2six_c(pose):
    """ Convert a pose tensor with rotation matrix representation (..., 3 , 3))
        into the 6D representation (..., 6)
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""

    out_shape = list(pose.shape[:-1])
    out_shape[-1] = 6

    # keeping only the first 2 columns
    a1 = pose[..., :, 0]
    a2 = pose[..., :, 1]

    a3 = a1 - \
        torch.tensor([1, 0, 0], requires_grad=False).repeat(
            [*a1.shape[:-1], 1]).to(a1.device)
    a4 = a2 - \
        torch.tensor([0, 1, 0], requires_grad=False).repeat(
            [*a2.shape[:-1], 1]).to(a2.device)
    return torch.cat((a3, a4), -1)


def six_c2rotmat(pose, eps=1e-6):
    """ Convert a pose tensor with 6D representation (..., 6)
        into the rotation matrix representation (..., 9)
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""
    checkTensor(pose)
    if pose.shape[-1:] != (6,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,6) got {}".format(pose.shape))

    x_raw = pose[..., 0:3]
    y_raw = pose[..., 3:6]

    x_raw2 = x_raw + torch.tensor([1, 0, 0], requires_grad=False).repeat([
        *x_raw.shape[:-1], 1]).to(x_raw.device)
    y_raw2 = y_raw + torch.tensor([0, 1, 0], requires_grad=False).repeat([
        *y_raw.shape[:-1], 1]).to(y_raw.device)

    x = x_raw2 / (x_raw2.norm(dim=-1, p='fro', keepdim=True) + eps)
    z = cross_product(x, y_raw2)
    z = z / (z.norm(dim=-1, p='fro', keepdim=True) + eps)
    y = cross_product(z, x)

    return torch.stack((x, y, z), -1)


def rotmat2aa(pose):
    """ Convert a pose tensor with rotation matrix representation (..., 3, 3)
        into an axis angle representation (..., 3 )"""
    checkTensor(pose)
    if pose.shape[-2:] != (3, 3):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3,3) got {}".format(pose.shape))

    return quat2aa(rotmat2quat(pose))


def aa2six(pose):
    """ Convert a pose tensor with axis-angle representation (..., 3))
        into a 6D representation (..., 6 )
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""

    return rotmat2six(aa2rotmat(pose))


def six2aa(pose):
    """ Convert a pose tensor with 6D representation (..., 6)
        into an axis angle representation (...,3 )
        Representation presented by Yi Zhou et Al (https://arxiv.org/abs/1812.07035)"""
    return rotmat2aa(six2rotmat(pose))


def aa2invmat(pose):
    """ Convert a pose tensor with axis-angle representation (..., 3)
        into the inverse rotation matrix representation (faster than aa2rotmat and then inverse each rotmat)"""

    checkTensor(pose)
    if pose.shape[-1:] != (3,):
        raise InvalidShapeException(
            "Invalid pose shape, expected (...,3) got {}".format(pose.shape))

    angle, x, y, z = extractAxisAngle(pose)

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rotMatCoord(-angle, x, y, z)

    invmat = torch.cat((r11, r12, r13, r21, r22, r23, r31, r32, r33), -1)

    shape = list(invmat.shape)
    shape[-1] = 3
    shape.append(3)

    return invmat.view(shape)


def conjugate_quat(q):

    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def multiply_quat(q1, q2):

    q3 = q1.clone()
    q3[..., 0] = q1[..., 0]*q2[..., 0] - \
        (q1[..., 1]*q2[..., 1]+q1[..., 2]*q2[..., 2]+q1[..., 3]*q2[..., 3])
    q3[..., 1] = q1[..., 0]*q2[..., 1] + q1[..., 1] * \
        q2[..., 0] + q1[..., 2]*q2[..., 3] - q1[..., 3]*q2[..., 2]
    q3[..., 2] = q1[..., 0]*q2[..., 2] - q1[..., 1] * \
        q2[..., 3] + q1[..., 2]*q2[..., 0] + q1[..., 3]*q2[..., 1]
    q3[..., 3] = q1[..., 0]*q2[..., 3] + q1[..., 1] * \
        q2[..., 2] - q1[..., 2]*q2[..., 1] + q1[..., 3]*q2[..., 0]

    return q3


def ln_quat(q):

    r = q[..., 1:].norm(dim=-1, keepdim=True)
    t = torch.zeros(r.shape, dtype=r.dtype, device=r.device)

    t[r > 0.00001] = torch.atan2(
        r[r > 0.00001], q[..., 0:1][r > 0.00001])/r[r > 0.00001]

    lnq = q.clone()

    lnq[..., 0:1] = torch.log(q.norm(dim=-1, keepdim=True))
    lnq[..., 1:] = t*q[..., 1:]

    return lnq


def exp_quat(q):

    r = q[..., 1:].norm(dim=-1, keepdim=True)
    et = torch.exp(q[..., 0:1])
    s = torch.zeros(r.shape, dtype=r.dtype, device=r.device)
    s[r > 0.00001] = et[r > 0.00001]*torch.sin(r[r > 0.00001])/r[r > 0.00001]

    expq = q.clone()
    expq[..., 0:1] = et*torch.cos(r)
    expq[..., 1:] = s*q[..., 1:]

    return expq


def power_quat(q, t):
    return exp_quat(t*ln_quat(q))


def slerp(q1, q2, t):
    return multiply_quat(power_quat(multiply_quat(q2, conjugate_quat(q1)), t), q1)


def convertPose(pose, source, target):
    """ Convert a pose tensor with source_repr representation (bs x nof x (source_angle_rep_dimxnoc)
        into the target representation (bs x nof x (target_angle_rep_dimxnoc))
        Only a few combination have been implemented yet"""
    checkTensor(pose)

    p = None

    source_repr = source["name"]
    target_repr = target["name"]

    source_dim = source["dim"]
    target_dim = target["dim"]

    out_shape = list(pose.shape)

    out_shape[-1] = out_shape[-1] * target_dim // source_dim

    if source_repr == target_repr:
        return pose

    if source_repr == "six":
        p = pose.contiguous().view(-1, 6)
        if target_repr == "mat":
            p = six2rotmat(p)
        elif target_repr == "aa":
            p = six2aa(p)
        elif target_repr == "quat":
            p = rotmat2quat(six2rotmat(p))
        elif target_repr == "euler":
            p = t3d.mat2euler(six2rotmat(p), "sxyz")

    elif source_repr == "aa":
        p = pose.contiguous().view(-1, 3)
        if target_repr == "mat":
            p = aa2rotmat(p)
        elif target_repr == "quat":
            p = aa2quat(p)
        elif target_repr == "six":
            p = aa2six(p)
        elif target_repr == "six_c":
            p = rotmat2six_c(aa2rotmat(p))
        elif target_repr == "invmat":
            p = aa2invmat(p)
        elif target_repr == "euler":
            p = t3d.mat2euler(aa2rotmat(p), "sxyz")

    elif source_repr == "quat":
        p = pose.contiguous().view(-1, 4)
        if target_repr == "aa":
            p = quat2aa(p)
        if target_repr == "mat":
            p = quat2rotmat(p)
        if target_repr == "invmat":
            p = aa2invmat(quat2aa(p))
        if target_repr == "six":
            p = rotmat2six(quat2rotmat(p))

    elif source_repr == "mat":
        p = pose.contiguous().view(-1, 3, 3)
        if target_repr == "aa":
            p = rotmat2aa(p)
        elif target_repr == "six":
            p = rotmat2six(p)
        elif target_repr == "six_c":
            p = rotmat2six_c(p)
        elif target_repr == "quat":
            p = rotmat2quat(p)
        elif target_repr == "euler":
            p = t3d.mat2euler(p, "sxyz")

    elif(source_repr == "six_c"):
        p = pose.contiguous().view(-1, 6)
        if target_repr == "mat":
            p = six_c2rotmat(p)
        elif target_repr == "aa":
            p = rotmat2aa(six_c2rotmat(p))

    elif(source_repr == "euler"):
        p = pose.contiguous().view(-1, 3)
        if target_repr == "aa":
            p = rotmat2aa(t3d.euler2mat(p, "sxyz"))
        elif target_repr == "mat":
            p = t3d.euler2mat(p, "sxyz")
        elif target_repr == "six":
            p = rotmat2six(t3d.euler2mat(p, "sxyz"))

    if(p is None):
        raise UnknownConversionException(
            "Sorry, this conversion was not implemented")
    else:

        return p.contiguous().view(out_shape)
