import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as eul
import imu_calibration as ic
import quarernion_calculation as qc
import orientation_estimate as oe
import torch

# Start counting time when program starts
import time
t0 = time.perf_counter()

# Loading and Preparing Data
dataset = '8'
cam_data = ld.get_data(dataset)[0]
imu_data = ld.get_data(dataset)[0]
vic_data = ld.get_data(dataset)[1]

# Defining how many samples available
num_sample_imu = imu_data.shape[1]
num_sample_vic = vic_data['ts'][0].shape[0]

# Calibrate the IMU data based on the first few seconds (frequency approximately 100Hz)
calibrated_imu = ic.imu_calibration(imu_data, 300)
calibrated_imu = torch.tensor(calibrated_imu)

# Setting up the first quaternion and the list to keep track of the predicted quaternions
q_t = torch.tensor([1,0,0,0])
all_q = [q_t]

# Calculating the predicted quaternions
for i in range(num_sample_imu - 1):
    tau = imu_data[0, i+1] - imu_data[0, i]
    q_tp1 = qc.motion_model(q_t, tau, calibrated_imu[:, i])
    all_q.append(q_tp1)
    q_t = q_tp1
q_torch = torch.stack(all_q)

# Prepare data before optimization and after optimization for later plotting
predicted_q = q_torch
predicted_q_opt = oe.gradient_descent(q_torch, calibrated_imu)

# Converting quaternion values to rotation matrices, then convert to roll/pitch/yaw values in degrees
predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q])
predicted_angles_rad = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices])
predicted_roll =    np.degrees(np.unwrap(predicted_angles_rad[:, 0]))
predicted_pitch =   np.degrees(np.unwrap(predicted_angles_rad[:, 1]))
predicted_yaw =     np.degrees(np.unwrap(predicted_angles_rad[:, 2]))

# Converting optimized quanternion values to rotation matrices, and then to roll/pitch/yaw values in degrees
predicted_rotation_matrices_opt = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q_opt])
predicted_angles_rad_opt = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices_opt])
predicted_roll_opt =    np.degrees(np.unwrap(predicted_angles_rad_opt[:, 0]))
predicted_pitch_opt =   np.degrees(np.unwrap(predicted_angles_rad_opt[:, 1]))
predicted_yaw_opt =     np.degrees(np.unwrap(predicted_angles_rad_opt[:, 2]))

# Converting rotation matrices got from VICON data and converting to roll/pitch/yaw values in degrees
true_rotation_matrices = vic_data['rots']
true_angles_rad = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_sample_vic)])
true_roll  =    np.degrees(np.unwrap(true_angles_rad[:, 0]))
true_pitch =    np.degrees(np.unwrap(true_angles_rad[:, 1]))
true_yaw   =    np.degrees(np.unwrap(true_angles_rad[:, 2]))

# Get the timeline of IMU and VICON data, and align them to avoid drifting in plotting
imu_time = imu_data[0]
vic_time = vic_data['ts'][0]
aligned_true_roll  = np.interp(imu_time, vic_time, true_roll)
aligned_true_pitch = np.interp(imu_time, vic_time, true_pitch)
aligned_true_yaw   = np.interp(imu_time, vic_time, true_yaw)

# Ending time after all calculations
t1 = time.perf_counter()
print(f"Time Elapsed: {t1 - t0:.6f} s")

# Plotting out values
# imu_rel -> IMU relative time: used to make the plotting looks better
imu_rel  = imu_time - imu_data[0][0]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,6), sharex=True)

ax1.plot(imu_rel, predicted_roll, label='Predicted Roll', color = 'r')
ax1.plot(imu_rel, aligned_true_roll, label='True Roll', color = 'b')
ax1.plot(imu_rel, predicted_roll_opt, label='Predicted Roll Optimized', color = 'g')
ax1.legend()

ax2.plot(imu_rel, predicted_pitch, label='Predicted Pitch', color = 'r')
ax2.plot(imu_rel, aligned_true_pitch, label='True Pitch', color = 'b')
ax2.plot(imu_rel, predicted_pitch_opt, label='Predicted Pitch Optimized', color = 'g')
ax2.legend()

ax3.plot(imu_rel, predicted_yaw, label='Predicted Yaw', color = 'r')
ax3.plot(imu_rel, aligned_true_yaw, label='True Yaw', color = 'b')
ax3.plot(imu_rel, predicted_yaw_opt, label='Predicted Yaw Optimized', color = 'g')
ax3.legend()

plt.show()