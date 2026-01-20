import numpy as np

def multiplication(q1, q2):
    # Source: https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def exponential(q):
    qs, qv1, qv2, qv3 = q
    qv = np.array([qv1, qv2, qv3])
    norm = np.linalg.norm(qv)

    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    vector_part = np.sin(norm) * qv / norm
    scalar_part = np.cos(norm)
    return np.array([scalar_part, vector_part[0], vector_part[1], vector_part[2]])

def motion_model(q_hat, time_stamp, imu_data):
    omega = np.array([imu_data[4], imu_data[5], imu_data[6]])
    vector_part = time_stamp * omega / 2
    exponential_imu = exponential(np.array([0, vector_part[0], vector_part[1], vector_part[2]]))
    new_q = multiplication(q_hat, exponential_imu)
    return new_q

def hat_mapping(mat):
    output_mat = np.zeros((3, 3))
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
    eye3 = np.eye(3)
    qs = q[0]
    qv = q[1:]
    second_term = qs * eye3 + hat_mapping(qv)
    eq = np.concatenate((-qv.reshape(3, 1), second_term), axis=1)
    return eq

def g(q):
    eye3 = np.eye(3)
    qs = q[0]
    qv = q[1:]
    second_term = qs * eye3 - hat_mapping(qv)
    eq = np.concatenate((-qv.reshape(3, 1), second_term), axis=1)
    return eq