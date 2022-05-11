import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import numpy as np

base_color = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/pose_graph/color_img/"
base_depth = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/pose_graph/depth/"
projection_vis_dir = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/projection_vis"
proj_dir = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1/projection_sample"

projected_flist = glob(os.path.join(proj_dir, "*.png"))
projected_flist = sorted(projected_flist)
fig = plt.figure()
for proj_f in tqdm(projected_flist):
    ref_frame_idx = proj_f.split("/")[-1].split("_")[0]
    ref_fname = os.path.join(base_color, "{}_image.png".format(ref_frame_idx))
    ref_im = plt.imread(ref_fname)
    target_im = plt.imread(proj_f)
    ax = plt.subplot(121)
    plt.imshow(ref_im)
    plt.axis(False)
    ax2 = plt.subplot(122)
    plt.imshow(target_im)
    plt.axis(False)
    plt.savefig(os.path.join(projection_vis_dir, proj_f.split("/")[-1]), bbox_inches='tight',
                pad_inches=0)
    plt.cla()
