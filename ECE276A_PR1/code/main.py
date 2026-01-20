import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as eul
import imu_calibration as ic
import quarernion_calculation as qc


q_t = np.array([1, 0, 0, 0])
all_qs = []

imu_data = ld.imud
vic_data = ld.vicd
num_samples = min(imu_data.shape[1], vic_data['rots'].shape[2])
calibrated_imu = ic.imu_calibration(imu_data)

# Calculating the predicted quaternions
for i in range(num_samples - 1):
    tau = imu_data[0, i+1] - imu_data[0, i]
    q_tp1 = qc.motion_model(q_t, tau, calibrated_imu[:, i])
    all_qs.append(q_tp1)
    q_t = q_tp1

predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q) for q in all_qs])
predicted_angles = np.array([eul.mat2euler(R) for R in predicted_rotation_matrices])
true_angles = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_samples)])

predicted_angles_deg = np.degrees(np.unwrap(predicted_angles, axis=0))
true_angles_deg = np.degrees(np.unwrap(true_angles, axis=0))

predicted_roll = predicted_angles_deg[:, 0]
predicted_pitch = predicted_angles_deg[:, 1]
predicted_yaw = predicted_angles_deg[:, 2]

true_roll = true_angles_deg[:, 0]
true_pitch = true_angles_deg[:, 1]
true_yaw = true_angles_deg[:, 2]

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

# print(predicted_angles)
# print(true_angles)