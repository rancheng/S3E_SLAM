import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
import cv2

from tqdm import tqdm
import natsort

out_root = "/mnt/Data/Shared/data_generate_2/vis_patchs"
data_root = "/mnt/Data/Shared/data_generate_2/torch_data"
data_root_parent = Path(data_root).parent
patch_flist = glob.glob(os.path.join(data_root, "*.h5"))
patch_flist = sorted(patch_flist)
patch_list = []
image_i = 1773
image_j = 1979
h5_name_i = patch_flist[image_i]
h5_name_j = patch_flist[image_j]
data_collection = {}
with h5py.File(h5_name_i) as f:
    data_collection['data_i'] = f['torch'][0, :]
with h5py.File(h5_name_j) as f:
    data_collection['data_j'] = f['torch'][0, :]
image_i_0 = data_collection['data_i'][0]
for i in range(data_collection['data_i'].shape[0]):
    if not os.path.exists(os.path.join(out_root, "depth", "ref")):
        Path(os.path.join(out_root, "depth", "ref")).mkdir(parents=True, exist_ok=True)
    colormap = plt.get_cmap('jet')
    heatmap = (colormap(data_collection['data_i'][i, :, :, 3]) * 255).astype(np.uint8)[:, :, :3]
    im = PIL.Image.fromarray(heatmap)
    im.save(os.path.join(out_root, "depth", "ref", "{}".format(i).zfill(6) + ".png"))
    if not os.path.exists(os.path.join(out_root, "color", "ref")):
        Path(os.path.join(out_root, "color", "ref")).mkdir(parents=True, exist_ok=True)
    im = cv2.cvtColor(data_collection['data_i'][i, :, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    im = PIL.Image.fromarray(im)
    im.save(os.path.join(out_root, "color", "ref", "{}".format(i).zfill(6) + ".png"))
for i in range(data_collection['data_j'].shape[0]):
    if not os.path.exists(os.path.join(out_root, "depth", "cur")):
        Path(os.path.join(out_root, "depth", "cur")).mkdir(parents=True, exist_ok=True)
    colormap = plt.get_cmap('jet')
    heatmap = (colormap(data_collection['data_j'][i, :, :, 3]) * 255).astype(np.uint8)[:, :, :3]
    im = PIL.Image.fromarray(heatmap)
    im.save(os.path.join(out_root, "depth", "cur", "{}".format(i).zfill(6) + ".png"))
    if not os.path.exists(os.path.join(out_root, "color", "cur")):
        Path(os.path.join(out_root, "color", "cur")).mkdir(parents=True, exist_ok=True)
    im = cv2.cvtColor(data_collection['data_j'][i, :, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    im = PIL.Image.fromarray(im)

    im.save(os.path.join(out_root, "color", "cur", "{}".format(i).zfill(6) + ".png"))
    # img_patch = image_i[i, :,:,:3].astype(np.uint8)
    # depth_patch = image_i[i, :, :, 3]
    # colormap = plt.get_cmap('inferno')
    # heatmap = (colormap(depth_patch) * 2 ** 16).astype(np.uint16)[:, :, :3]

# plt.imshow(image_i_0)
# plt.xlabel("frame j")
# plt.ylabel("frame i")
# plt.colorbar()
# plt.title("pixel-wise IoU heat map")
# plt.show()