import numpy as np

def imu_calibration_gyro(imu_data, calib_range):
    # Gyro calibration
    w_x = imu_data[4]
    w_y = imu_data[5]
    w_z = imu_data[6]

    vref = 3300
    sensitivity_w = 3.33*(180/np.pi)
    scale_factor_w = (vref/1023)/sensitivity_w

    bias_wx = np.mean(w_x[:calib_range])
    bias_wy = np.mean(w_y[:calib_range])
    bias_wz = np.mean(w_z[:calib_range])

    w_x = (w_x - bias_wx)*scale_factor_w
    w_y = (w_y - bias_wy)*scale_factor_w
    w_z = (w_z - bias_wz)*scale_factor_w
    
    return(w_x, w_y, w_z)


def imu_calibration_accel(imu_data, calib_range):
    a_x = imu_data[1]
    a_y = imu_data[2]
    a_z = imu_data[3]

    vref = 3300
    sensitivity_a = 330
    scale_factor_a = (vref/1023)/sensitivity_a

    bias_ax = np.mean(a_x[:calib_range])
    bias_ay = np.mean(a_y[:calib_range])
    bias_az = (np.mean(a_z[:calib_range])-1/scale_factor_a)


    a_x = (a_x - bias_ax)*scale_factor_a
    a_y = (a_y - bias_ay)*scale_factor_a
    a_z = (a_z - bias_az)*scale_factor_a

    return(a_x, a_y, a_z)

def imu_calibration(imu_data, calib_range):
    a_x, a_y, a_z = imu_calibration_accel(imu_data, calib_range)
    w_x, w_y, w_z = imu_calibration_gyro(imu_data, calib_range)
    calibrated_imu = np.array([imu_data[0], a_x, a_y, a_z, w_x, w_y, w_z])
    return calibrated_imu