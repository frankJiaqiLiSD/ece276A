from turtle import color
import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as eul
import imu_calibration as ic
import quarernion_calculation as qc
import orientation_estimate as oe
import torch

#--------start timing---
import time
t0 = time.perf_counter()
#-----------------------

q_t = torch.tensor([1,0,0,0])
all_qs = [q_t]

imu_data = ld.imud
vic_data = ld.vicd
num_sample_imu = imu_data.shape[1]
num_sample_vic = vic_data['ts'][0].shape[0]


calibrated_imu = ic.imu_calibration(imu_data, 300)
calibrated_imu = torch.tensor(calibrated_imu)

imu_time = imu_data[0]
imu_rel  = imu_time - imu_data[0][0]
vic_time = vic_data['ts'][0]


# Calculating the predicted quaternions
for i in range(num_sample_imu - 1):
    tau = imu_data[0, i+1] - imu_data[0, i]
    q_tp1 = qc.motion_model(q_t, tau, calibrated_imu[:, i])
    all_qs.append(q_tp1)
    q_t = q_tp1
q_torch = torch.stack(all_qs)

predicted_q_opt = oe.gradient_descent(q_torch, calibrated_imu)
predicted_q = q_torch

# predicted angles for plotting
predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q])
predicted_angles_rad = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices])
predicted_angles_deg = np.degrees(predicted_angles_rad)

predicted_roll =    predicted_angles_deg[:, 0].round(2)
predicted_pitch =   predicted_angles_deg[:, 1].round(2)
predicted_yaw =     predicted_angles_deg[:, 2].round(2)


# true angles for plotting
true_rotation_matrices = vic_data['rots']
true_angles_rad = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_sample_vic)])
true_angles_deg = np.degrees(true_angles_rad)


true_roll =    true_angles_deg[:, 0]
true_pitch =   true_angles_deg[:, 1]
true_yaw =     true_angles_deg[:, 2]

aligned_true_roll  = np.interp(imu_time, vic_time, true_roll)
aligned_true_pitch = np.interp(imu_time, vic_time, true_pitch)
aligned_true_yaw   = np.interp(imu_time, vic_time, true_yaw)


# optimized for plotting
predicted_rotation_matrices_opt = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q_opt])
predicted_angles_rad_opt = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices_opt])
predicted_angles_deg_opt = np.degrees(predicted_angles_rad_opt)

predicted_roll_opt =    predicted_angles_deg_opt[:, 0].round(2)
predicted_pitch_opt =   predicted_angles_deg_opt[:, 1].round(2)
predicted_yaw_opt =     predicted_angles_deg_opt[:, 2].round(2)


#-----------end timing------------
t1 = time.perf_counter()
print(f"Elapsed: {t1 - t0:.6f} s")
#---------------------------------


# plotting out compare diagram
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30,18), sharex=True)

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