import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as eul
import imu_calibration as ic
import quarernion_calculation as qc
import orientation_estimate as oe
import torch


def get_nearest_index(target_time, timestamp_array):
    idx = (np.abs(timestamp_array - target_time)).argmin()
    return idx

q_t = torch.tensor([1,0,0,0])
all_qs = [q_t]

imu_data = ld.imud
vic_data = ld.vicd
num_samples = min(imu_data.shape[1], vic_data['rots'].shape[2])
calibrated_imu = ic.imu_calibration(imu_data)
calibrated_imu = torch.tensor(calibrated_imu)

# Calculating the predicted quaternions
for i in range(num_samples - 1):
    tau = imu_data[0, i+1] - imu_data[0, i]
    q_tp1 = qc.motion_model(q_t, tau, calibrated_imu[:, i])
    all_qs.append(q_tp1)
    q_t = q_tp1
q_torch = torch.stack(all_qs)
# print(calibrated_imu.shape[1])
# print(q_torch.shape[0])
# print(oe.cost_function(q_torch,calibrated_imu).item())
# print(oe.gradient(q_torch,calibrated_imu))

# predicted_q = oe.gradient_descent(q_torch, calibrated_imu)
predicted_q = q_torch
# print(predicted_q)

# print(vic_data['ts'][0])
# print(imu_data[0])

# time_1 = np.array(vic_data['ts'][0][0])
# time_2 = np.array(imu_data[0][-1])
# time_3 = np.array(vic_data['ts'][0][0])
# time_4 = np.array(imu_data[0][-1])

# print(time_1 - time_2)
# print(time_3 - time_4)

# predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q) for q in all_qs])
predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q])
predicted_angles = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices])


# true_angles = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_samples)])

# predicted_angles_deg = np.degrees(np.unwrap(predicted_angles, axis=0))
# true_angles_deg = np.degrees(np.unwrap(true_angles, axis=0))

predicted_angles_deg = np.degrees(predicted_angles)
# true_angles_deg = np.degrees(true_angles)

predicted_roll = predicted_angles_deg[:, 0]
predicted_pitch = predicted_angles_deg[:, 1]
predicted_yaw = predicted_angles_deg[:, 2]

# true_roll = true_angles_deg[:, 0]
# true_pitch = true_angles_deg[:, 1]
# true_yaw = true_angles_deg[:, 2]

# === NEW ALIGNMENT CODE START ===

# 1. Get all timestamps
# IMU timestamps (corresponding to your predicted_angles)
imu_ts = imu_data[0, :num_samples] 

# Vicon timestamps (ensure it's a 1D array)
# Check shape: vic_data['ts'] might be (1, N) or (N,)
vic_ts = vic_data['ts']
if vic_ts.ndim > 1:
    vic_ts = vic_ts.flatten()

# 2. Pre-calculate ALL Vicon Euler angles (ground truth source)
# We need the full set to look up the correct values
num_vic_samples = vic_data['rots'].shape[2]
all_vicon_angles = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_vic_samples)])
all_vicon_deg = np.degrees(all_vicon_angles) # Shape: (N_vicon, 3)

# 3. Align Vicon data to IMU timestamps
aligned_true_roll = []
aligned_true_pitch = []
aligned_true_yaw = []

print("Aligning timestamps... this may take a moment.")
for t in imu_ts:
    # Find the index in Vicon data closest to the current IMU timestamp
    nearest_idx = get_nearest_index(t, vic_ts)
    
    # Grab the ground truth from that index
    r, p, y = all_vicon_deg[nearest_idx]
    
    aligned_true_roll.append(r)
    aligned_true_pitch.append(p)
    aligned_true_yaw.append(y)

# Convert to numpy arrays for plotting
true_roll = np.array(aligned_true_roll)
true_pitch = np.array(aligned_true_pitch)
true_yaw = np.array(aligned_true_yaw)

# === NEW ALIGNMENT CODE END ===



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

ax1.plot(predicted_roll, label='Predicted Roll')
ax1.plot(true_roll, label='True Roll')
ax1.legend()

ax2.plot(predicted_pitch, label='Predicted Pitch')
ax2.plot(true_pitch, label='True Pitch')
ax2.legend()

ax3.plot(predicted_yaw, label='Predicted Yaw')
ax3.plot(true_yaw, label='True Yaw')
ax3.legend()

plt.show()