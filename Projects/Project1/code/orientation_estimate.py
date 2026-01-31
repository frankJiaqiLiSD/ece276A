import quarernion_calculation as qc
import torch
import os
import matplotlib.pyplot as plt

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
    cost_wrapper = lambda q: cost_function(q, imu_data)
    grad_tensor = torch.autograd.functional.jacobian(cost_wrapper, q_array)    
    return grad_tensor

def gradient_descent(q_array, imu_data, num_of_epoch, step_length, dataset):
    q = q_array.detach().clone()
    loss = []
    prev_loss = cost_function(q, imu_data).item()

    for _ in range(num_of_epoch):
        grad = gradient(q, imu_data)
        new_q_array = q - step_length*grad
        norm = torch.norm(new_q_array, dim=1, keepdim=True)
        new_q_array = new_q_array / norm
        new_loss = cost_function(new_q_array, imu_data).item()

        if new_loss < prev_loss:
            q = new_q_array
            prev_loss = new_loss
        
        loss.append(prev_loss)

    plt.figure()
    plt.plot(loss, color = 'b')
    plt.title(f"Loss for Dataset {dataset}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs("img", exist_ok=True)
    save_path = f"img/Dataset {str(dataset)}_loss.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return q.detach()