import torch

def multiplication(q1, q2):
    # Source: https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q = torch.stack([w, x, y, z])
    return q

def exponential(q):
    qs, qv1, qv2, qv3 = q
    qv = torch.stack([qv1, qv2, qv3])
    norm = torch.norm(qv)
    
    if norm < 1e-8:
        return torch.tensor([torch.exp(qs), 0, 0, 0])

    vector_part = torch.sin(norm) * qv / norm
    scalar_part = torch.cos(norm)
    exponential = torch.stack([scalar_part, vector_part[0], vector_part[1], vector_part[2]])
    return exponential

def motion_model(q, tau, imu_data):
    omega = torch.stack([imu_data[4], imu_data[5], imu_data[6]])
    vector_part = tau * omega / 2
    zero_tensor = torch.tensor(0.0)
    exp_input = torch.stack([zero_tensor, vector_part[0], vector_part[1], vector_part[2]])
    exponential_imu = exponential(exp_input)
    new_q = multiplication(q, exponential_imu)
    return new_q

def hat_mapping(mat):
    output_mat = torch.zeros(3, 3)
    output_mat[0, 1] = -mat[2]
    output_mat[0, 2] = mat[1]
    output_mat[1, 0] = mat[2]
    output_mat[1, 2] = -mat[0]
    output_mat[2, 0] = -mat[1]
    output_mat[2, 1] = mat[0]
    return output_mat

def quaternion_to_rotation_matrix(q):
    eq = e(q)
    gq = g(q)
    rq = eq @ gq.T
    return rq

def e(q):
    eye3 = torch.eye(3)
    qs = q[0]
    qv = q[1:]
    second_term = qs * eye3 + hat_mapping(qv)
    eq = torch.cat((-qv.view(3, 1), second_term), dim=1)
    return eq

def g(q):
    eye3 = torch.eye(3)
    qs = q[0]
    qv = q[1:]
    second_term = qs * eye3 - hat_mapping(qv)
    gq = torch.cat((-qv.view(3, 1), second_term), dim=1)
    return gq

def observation_model(q):
    gravity_ref = torch.tensor([0.0, 0.0, 0.0, 1.0])
    q_inv = inverse(q)
    q_inv_ref = multiplication(q_inv, gravity_ref)
    observation = multiplication(q_inv_ref, q)
    return observation

def quaternion_log(q):
    qs = q[0]
    qv = q[1:]
    norm_qv = torch.norm(qv)
    norm_q = torch.norm(q)

    if norm_qv < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    scalar_part = torch.log(norm_q)
    vector_part = (qv / norm_qv) * (torch.acos(qs / norm_q))

    return torch.cat((scalar_part.unsqueeze(0), vector_part))

def inverse(q):
    q_bar = torch.cat((q[0:1], -q[1:]))
    norm_q = torch.linalg.norm(q)
    return q_bar / (norm_q**2)

def quaternion_norm(q):
    qs = q[0]
    qv = q[1:]
    qs_sq = qs ** 2
    transpose_multiply = torch.dot(qv, qv)
    norm = torch.sqrt(qs_sq + transpose_multiply)

    return norm