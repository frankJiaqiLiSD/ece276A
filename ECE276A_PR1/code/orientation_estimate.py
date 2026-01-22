import quarernion_calculation as qc
import torch

def cost_function(q_array, imu_data):
    motion_model_error = 0
    observation_model_error = 0
    num_samples = q_array.shape[0]

    for i in range(0, num_samples-1):
        tau = imu_data[0, i+1] - imu_data[0, i]

        q_pred = qc.motion_model(q_array[i], tau, imu_data[:, i])
        q_next_inv = qc.inverse(q_array[i+1])
        single_motion_model_error = 2*qc.quaternion_log(qc.multiplication(q_next_inv, q_pred))
        motion_model_error += 0.5 * (torch.linalg.norm(single_motion_model_error)**2)
    for j in range(1, num_samples):
        ax = imu_data[1, j]
        ay = imu_data[2, j]
        az = imu_data[3, j]

        single_acceleration = torch.stack([torch.tensor(0.0), ax, ay, az])
        single_observation_model_error = single_acceleration - qc.observation_model(q_array[j])
        observation_model_error += 0.5 * (torch.linalg.norm(single_observation_model_error)**2)
    
    return motion_model_error + observation_model_error


def gradient(q_array, imu_data):
    # cost_wrapper = lambda q: cost_function(q, imu_data)
    # grad_tensor = torch.autograd.functional.jacobian(cost_wrapper, q_array)
    cost = cost_function(q_array, imu_data)           
    (grad,) = torch.autograd.grad(cost, q_array)     
    return grad

def gradient_descent(q_array, imu_data):
    # num_of_epoch = 2
    # step_length = 0.1
    # q = q_array.detach().clone()

    # for _ in range(num_of_epoch):
    #     grad = gradient(q, imu_data)
    #     new_q_array = q + step_length*grad
    #     norm = torch.norm(new_q_array, dim=1, keepdim=True)
    #     q = new_q_array / norm
    
    num_of_epoch = 100
    step_length = 0.01
    q = q_array.detach().clone()

    for _ in range(num_of_epoch):
        q = q.detach().clone().requires_grad_(True)
        grad_dir = gradient(q, imu_data)
        with torch.no_grad():
            q = q - step_length * grad_dir
            q = q / torch.norm(q, dim=1, keepdim=True)

    return q.detach()