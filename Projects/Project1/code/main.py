import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import transforms3d.euler as eul
import imu_calibration as ic
import quarernion_calculation as qc
import orientation_estimate as oe
import torch
import panorama as pa
import os
import time

def motion_predict(dataset):
    cam_data = ld.get_data(dataset)[0]
    imu_data = ld.get_data(dataset)[1]
    vic_data = ld.get_data(dataset)[2]
    imu_time = imu_data[0]

    vic_list = [1,2,3,4,5,6,7,8,9]
    cam_list = [1,2,8,9,10,11]

    # Defining how many samples available
    num_sample_imu = imu_data.shape[1]
    if int(dataset) in vic_list:
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
    num_of_epoch = 15
    step_length  = 0.1
    predicted_q = q_torch
    predicted_q_opt = oe.gradient_descent(q_torch, calibrated_imu, num_of_epoch, step_length, dataset)

    # Converting quaternion values to rotation matrices, then convert to roll/pitch/yaw values in degrees
    predicted_rotation_matrices = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q])
    predicted_angles_rad = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices])
    predicted_roll =    np.degrees(predicted_angles_rad[:, 0]).round(3)
    predicted_pitch =   np.degrees(predicted_angles_rad[:, 1]).round(3)
    predicted_yaw =     np.degrees(predicted_angles_rad[:, 2]).round(3)

    # Converting optimized quanternion values to rotation matrices, and then to roll/pitch/yaw values in degrees
    predicted_rotation_matrices_opt = np.array([qc.quaternion_to_rotation_matrix(q).detach().numpy() for q in predicted_q_opt])
    predicted_angles_rad_opt = np.array([eul.mat2euler(np.asarray(R)) for R in predicted_rotation_matrices_opt])
    predicted_roll_opt =    np.degrees(predicted_angles_rad_opt[:, 0]).round(3)
    predicted_pitch_opt =   np.degrees(predicted_angles_rad_opt[:, 1]).round(3)
    predicted_yaw_opt =     np.degrees(predicted_angles_rad_opt[:, 2]).round(3)

    if int(dataset) in vic_list:
        # Converting rotation matrices got from VICON data and converting to roll/pitch/yaw values in degrees
        true_angles_rad = np.array([eul.mat2euler(vic_data['rots'][:, :, i]) for i in range(num_sample_vic)])
        true_roll  =    np.degrees(true_angles_rad[:, 0]).round(3)
        true_pitch =    np.degrees(true_angles_rad[:, 1]).round(3)
        true_yaw   =    np.degrees(true_angles_rad[:, 2]).round(3)

        # Get the timeline of IMU and VICON data, and align them to avoid drifting in plotting
        imu_time = imu_data[0]
        vic_time = vic_data['ts'][0]
        aligned_true_roll  = np.interp(imu_time, vic_time, true_roll)
        aligned_true_pitch = np.interp(imu_time, vic_time, true_pitch)
        aligned_true_yaw   = np.interp(imu_time, vic_time, true_yaw)

    if int(dataset) in cam_list:
        predicted_rotation_matrices_opt = predicted_rotation_matrices_opt.transpose(1,2,0)

        moving_data = {'ts': imu_time, 'rots': predicted_rotation_matrices_opt}
        coordinate = pa.calculate_panorama(cam_data, moving_data)

        pano_path = f"img/Dataset {str(dataset)}_panorama.png"
        pa.draw_panorama(coordinate, cam_data, pano_path)

    os.makedirs("img", exist_ok=True)
    # Plotting out values (with ground true data available)
    # imu_rel -> IMU relative time: used to make the plotting looks better
    imu_rel  = imu_time - imu_data[0][0]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,6), sharex=True)

    ax1.plot(imu_rel, predicted_roll, color = 'r')
    ax1.plot(imu_rel, predicted_roll_opt, color = 'g')
    if int(dataset) in vic_list:
        ax1.set_title("True Roll(blue) vs Estimated Roll(red) vs Optimized Estimated Roll(green) in Degrees")
        ax1.plot(imu_rel, aligned_true_roll, color = 'b')
    else:
        ax1.set_title("Estimated Roll(red) vs Optimized Estimated Roll(green) in Degrees")

    ax2.plot(imu_rel, predicted_pitch, color = 'r')
    ax2.plot(imu_rel, predicted_pitch_opt, color = 'g')
    if int(dataset) in vic_list:
        ax2.set_title("True Pitch(blue) vs Estimated Pitch(red) vs Optimized Estimated Pitch(green) in Degrees")
        ax2.plot(imu_rel, aligned_true_pitch, color = 'b')
    else:
        ax2.set_title("Estimated Pitch(red) vs Optimized Estimated Pitch(green) in Degrees")

    ax3.plot(imu_rel, predicted_yaw, color = 'r')
    ax3.plot(imu_rel, predicted_yaw_opt, color = 'g')
    if int(dataset) in vic_list:
        ax3.set_title("True Yaw(blue) vs Estimated Yaw(red) vs Optimized Estimated Yaw(green) in Degrees")
        ax3.plot(imu_rel, aligned_true_yaw, color = 'b')
    else:
        ax3.set_title("Estimated Yaw(red) vs Optimized Estimated Yaw(green) in Degrees")

    plot_path = f"img/Dataset {str(dataset)}_rpy.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    num_of_datasets = 11
    for dataset in range(1, num_of_datasets+1):
        t_start = time.time()
        motion_predict(str(dataset))
        t_end = time.time()
        elapsed = t_end - t_start
        print("Dataset {}/{} finished. Elapsed Time: {:.2f}s. ".format(dataset, num_of_datasets, elapsed))