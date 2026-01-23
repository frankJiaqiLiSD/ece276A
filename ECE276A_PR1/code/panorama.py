import load_data as ld
import numpy as np
import matplotlib.pyplot as plt

# Loading and Preparing Data
dataset = '1'
cam_data = ld.get_data(dataset)[0]['cam']
vic_data = ld.get_data(dataset)[2]['rots']

width = cam_data.shape[0]
length = cam_data.shape[1]

fov = 60*np.pi/180 # assumed value
cu = length/2
cv = width/2
f = ((length/2)/(np.tan(fov/2))).round()

# Converting from pixel frame to camera frame (inverse-intrinsic)
intrinsic_matrix = np.array([[f, 0, cu],
                             [0, f, cv],
                             [0, 0, 1]])
# print(cam_data[:,:,0,0])


W_pano = 1280
H_pano = 960
Panorama = np.zeros((H_pano, W_pano, 3), dtype=np.uint8)
image_0 = cam_data[:, :, :, 0]
# print(vic_data[:,:,0].shape)
for i in range(length):
    for j in range(width):
        pixel_mat = np.array([[i],[j],[1]])
        camera_mat = np.dot(np.linalg.inv(intrinsic_matrix), pixel_mat)
        world_mat = np.dot(vic_data[:,:,0], camera_mat)

        xw, yw, zw = world_mat
        norm = np.sqrt(xw**2 + yw**2 + zw**2)
        phi = np.arcsin(zw / norm)
        theta = np.arctan2(yw, xw)

        pano_x = int(((theta + np.pi) / (2*np.pi) * W_pano)[0])
        pano_y = 1-int(((phi + np.pi/2) / np.pi * H_pano)[0])
        
        Panorama[pano_y, pano_x, :] = image_0[j, i, :]

# print(cam_data[:,:,0,0].shape)
plt.imshow(Panorama.astype(np.uint8))
plt.show()
# print(intrinsic_matrix)
# print(length, width)