import load_data as ld
import numpy as np
import matplotlib.pyplot as plt

dataset = '2'
cam_data = ld.get_data(dataset)[0]['cam']   # Shape: 240*320*3*N
vic_data = ld.get_data(dataset)[2]['rots']  # Shape: 3*3*M
vic_time = ld.get_data(dataset)[2]['ts']    # Shape: 1*M
cam_time = ld.get_data(dataset)[0]['ts'][0] # Shape: 1*N

def align_time(cam_time, vic_data, vic_timestamps):
    aligned_rotation_matrix = np.zeros((3, 3, cam_time.shape[0]))
    i = 0
    for stamp in cam_time:
        idx = np.argmin(np.abs(stamp - vic_timestamps))
        aligned_rotation_matrix[:, :, i] = vic_data[:, :, idx]
        i += 1
    return aligned_rotation_matrix

aligned_rotation_matrix = align_time(cam_time, vic_data, vic_time)

fov_horizontal = 60/180*np.pi
fov_vertical   = 45/180*np.pi
pic_height, pic_width, _, N = cam_data.shape


'''
fsu: Horizontal (align with pic_width) Focal Length
fsv: Vertical (align with pic_height) Focal Length
cu: Horizontal Principle Point
cv: Vertical Principle Point
'''
fsu = int(pic_width / (2*np.tan((fov_horizontal/2))))   # fsu = 277
fsv = int(pic_height / (2*np.tan((fov_vertical/2))))    # fsv = 289
cu = pic_width / 2  # cu = 160
cv = pic_height / 2 # cv = 120


'''
Converting to optical frame: [Xo, Yo, Zo]^T
Skew factor is 0 here since the pixels are rectangular and coordinate axes are proportional
u: horizontal coordiates increment vector
v: vertical coordinates increment vector
'''
calib_mat = np.array([[fsu, 0  , cu],
                      [0,   fsv, cv],
                      [0,   0,   1 ]])
calib_mat_inv = np.linalg.inv(calib_mat)
u = np.arange(pic_width)
v = np.arange(pic_height)
X,Y = np.meshgrid(u,v)
pixel_coor_mat = np.stack([X.flatten(),
                          Y.flatten(),
                          np.ones(pic_width*pic_height)])
opt_coor_mat = calib_mat_inv @ pixel_coor_mat # 3*3 multiply 3*76800


'''
Finding Longitude(lambda) and Latitude(phi)
Converting to cartesian coordinate waiting to be rotated
'''
longitude = np.arctan2(opt_coor_mat[0], opt_coor_mat[2])
latitude  = np.arctan2(-opt_coor_mat[1], np.sqrt(opt_coor_mat[0]**2 + opt_coor_mat[2]**2))
sphere_coor_mat = np.stack([longitude, latitude])
x, y, z = np.cos(latitude) * np.sin(longitude), -np.sin(latitude), np.cos(latitude) * np.cos(longitude)
cat_coor_mat = np.stack([x,y,z])


'''
Rotating to world frame based on the rotation matrix provided
aligned_rotation_matrix: shape 3*3*N
cat_coor_mat: shape 3*76800
'''
R_body_from_opt = np.array([[ 0,  0,  1],
                            [-1,  0,  0],
                            [ 0, -1,  0]])

cat_body = R_body_from_opt @ cat_coor_mat
results_list = []

for i in range(N):
    rotated_data = aligned_rotation_matrix[:, :, i] @ cat_body
    results_list.append(rotated_data)

world_coor_mat = np.stack(results_list, axis=0)


'''
Converting back to spherical coordinate from world frame cartesian coordinate
longitude_conv: longitude converted from world frame
latidude_conv: latitude converted from world frame
'''
x = world_coor_mat[:, 0, :]
y = world_coor_mat[:, 1, :]
z = world_coor_mat[:, 2, :]

longitude_conv = np.arctan2(y, x)
latidude_conv  = np.arctan2(z, np.sqrt(x*x + y*y))

cyl_coor_mat = np.stack([longitude_conv, latidude_conv, np.ones(longitude_conv.shape)])


'''
Drawing to the canvas with width 2pi and height pi
'''
H_map = 1000
W_map = 2000
panorama = np.zeros((H_map, W_map, 3), dtype=np.uint8)

colors = cam_data.transpose(2, 3, 0, 1).reshape(3, N, -1) # Shape: 3*N*76800

flat_coords = cyl_coor_mat.reshape(3, -1)   # Shape: [X,Y,Z]^T*(N*76800)
flat_colors = colors.reshape(3, -1)         # Shape: [R,G,B]^T*(N*76800)
pixel_colors = flat_colors.T.astype(np.uint8)

longitudes = flat_coords[0, :]
latitudes  = flat_coords[1, :]

u_coords = ((longitudes + np.pi) / (2 * np.pi) * W_map).astype(np.int32)
v_coords = ((np.pi/2 - latitudes) / np.pi * (H_map - 1)).astype(np.int32)

# u_coords = np.clip(u_coords, 0, W_map - 1)
# v_coords = np.clip(v_coords, 0, H_map - 1)

panorama[v_coords, u_coords] = pixel_colors

plt.axis("off")
plt.imshow(panorama)
plt.show()