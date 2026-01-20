import numpy as np
import quarernion_calculation as qc
# import torch

def cost_function(q_array, imu_data):
    motion_model_error = 0
    observation_model_error = 0
    num_samples = q_array.shape[0]

    for i in range(0, num_samples-1):
        tau = imu_data[0, i+1] - imu_data[0, i]

        q_pred = qc.motion_model(q_array[i], tau, imu_data[:, i])
        q_next_inv = qc.inverse(q_array[i+1])
        single_motion_model_error = 2*qc.quaternion_log(qc.multiplication(q_next_inv, q_pred))
        motion_model_error += 0.5 * (np.linalg.norm(single_motion_model_error)**2)
    for j in range(1, num_samples):
        ax = imu_data[1, j]
        ay = imu_data[2, j]
        az = imu_data[3, j]

        single_acceleration = np.array([0, ax, ay, az])
        single_observation_model_error = single_acceleration - qc.observation_model(q_array[j])
        observation_model_error += 0.5 * (np.linalg.norm(single_observation_model_error)**2)
    
    return motion_model_error + observation_model_error
