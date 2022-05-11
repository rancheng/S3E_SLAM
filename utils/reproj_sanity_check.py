# Copyright 2021 Ran Cheng <ran.cheng2@mail.mcgill.ca>
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import transformations
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

pose_graph_base_dir = "/home/ran/Documents/D435i_pose_graph/pose_graph"
pose_fname = "vins_result_loop_tum.txt"
keypoints_dir = "key_points"
color_img_dir = "color_img"
depth_dir = "depth"

# load pose file
pose_data = np.genfromtxt(os.path.join(pose_graph_base_dir, pose_fname), dtype=float, delimiter=" ")

# load the image file based on pose
for i in range(len(pose_data)-1):
    cur_timestamp = pose_data[i, 0]
    cur_translation = pose_data[i, 1:4]
    cur_quaternion = pose_data[i, 4:8]
    next_timestamp = pose_data[i+1, 0]
    next_translation = pose_data[i+1, 1:4]
    next_quaternion = pose_data[i+1, 4:8]
    cur_rgb_image_fname = os.path.join(pose_graph_base_dir, color_img_dir, "{}_image.png".format(i))
    cur_rgb_image_array = cv2.imread(cur_rgb_image_fname)
    next_rgb_image_fname = os.path.join(pose_graph_base_dir, color_img_dir, "{}_image.png".format(i+1))
    next_rgb_image_array = cv2.imread(next_rgb_image_fname)
    