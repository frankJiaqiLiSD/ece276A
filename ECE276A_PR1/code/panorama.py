import load_data as ld
import numpy as np
import matplotlib.pyplot as plt

dataset = '2'
cam_data = ld.get_data(dataset)[0]['cam']
vic_data = ld.get_data(dataset)[2]['rots']
vic_time = ld.get_data(dataset)[2]['ts']
cam_time = ld.get_data(dataset)[0]['ts'][0]

pic_width = cam_data.shape[0]
pic_length = cam_data.shape[1]

fov = 60*np.pi/180 # assumed value
cu = pic_length/2
cv = pic_width/2
f = ((pic_width)/(2 * np.tan(fov/2))).round()

pano_width = pic_width*6
pano_length = pic_length*6
panorama = np.zeros((pano_width, pano_length, 3), dtype=np.uint8)

def align_time(cam_time, vic_data, vic_timestamps):
    aligned_rotation_matrix = np.zeros((3, 3, cam_time.shape[0]))
    i = 0
    for stamp in cam_time:
        idx = np.argmin(np.abs(stamp - vic_timestamps))
        aligned_rotation_matrix[:, :, i] = vic_data[:, :, idx]
        # print(vic_data[:, :, idx].shape)
        i += 1
    return aligned_rotation_matrix

'''
p_loc: Local position of pixels
p_wol: World Position
cu: Center of length
cv: Center of Width
pu: x coordinate of pano
pv: y coordinate of pano
'''

u_range = np.arange(pic_length)
v_range = np.arange(pic_width)
u_grid, v_grid = np.meshgrid(u_range, v_range)

x_local = u_grid - cu
y_local = v_grid - cv
z_local = np.full_like(x_local, f)

p_loc = np.stack([x_local, y_local, z_local])
norms = np.linalg.norm(p_loc, axis=0)
p_loc = p_loc / norms

aligned_rot_mat = align_time(cam_time, vic_data, vic_time)

for n in range(len(cam_time)):
    img = cam_data[:, :, :, n]
    single_rot_matrix = aligned_rot_mat[:, :, n]

#     p_wol = single_rot_matrix @ p_loc

#     phi = np.arctan2(p_wol[0, :], p_wol[2, :])
#     theta = np.arcsin(p_wol[1, :])

#     pu = int((pano_width * (phi + np.pi)/(2*np.pi))[0])
#     pv = int((pano_length * (theta + (np.pi/2))/(np.pi))[0])

#     panorama[pv, pu] = img.reshape(-1, 3)

# plt.figure(figsize=(15, 7))
# plt.imshow(panorama)
# plt.axis('off')
# plt.show()

print(p_loc.shape)

