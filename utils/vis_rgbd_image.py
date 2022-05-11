import matplotlib.pyplot as plt
import os
import numpy as np

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

base_color = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/pose_graph/color_img/"
base_depth = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/pose_graph/depth/"

target_img_idx = 2032
ref_img_idx = 1805

target_img_depth = "{}_depth.png".format(target_img_idx)
target_img_color = "{}_image.png".format(target_img_idx)
ref_img_depth = "{}_depth.png".format(ref_img_idx)
ref_img_color = "{}_image.png".format(ref_img_idx)

target_img_color_fname = os.path.join(base_color, target_img_color)
target_img_depth_fname = os.path.join(base_depth, target_img_depth)

ref_img_color_fname = os.path.join(base_color, ref_img_color)
ref_img_depth_fname = os.path.join(base_depth, ref_img_depth)

# read target frame data
target_color_im = plt.imread(target_img_color_fname)
target_depth_im = plt.imread(target_img_depth_fname)
# read reference frame data
ref_color_im = plt.imread(ref_img_color_fname)
ref_depth_im = plt.imread(ref_img_depth_fname)

# ref_color_im = ref_color_im[:, :, ::-1]
# target_color_im = target_color_im[:, :, ::-1]

target_depth_im[target_depth_im > 0.1] = 0.1
ref_depth_im[ref_depth_im > 0.1] = 0.1

fig = plt.figure()
plt.subplot(221)
plt.imshow(ref_color_im)
plt.axis(False)
plt.subplot(222)
plt.imshow(ref_depth_im, cmap='jet')
plt.axis(False)
plt.subplot(223)
plt.imshow(target_color_im)
plt.axis(False)
plt.subplot(224)
plt.imshow(target_depth_im, cmap='jet')
plt.axis(False)
plt.show()