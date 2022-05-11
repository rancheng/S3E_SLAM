import glob
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import natsort

data_root = "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_5/torch_data"
data_root_parent = Path(data_root).parent
out_iou_heatmap_fname = "iou_heatmap.npy"
iou_flist = glob.glob(os.path.join(data_root, "*.h5"))
iou_flist = sorted(iou_flist)
ious_list = []
for i in tqdm(range(len(iou_flist))):
    h5_name = iou_flist[i]
    with h5py.File(h5_name) as f:
        ious_list.append(f['ious'][:])
ious_arr = np.array(ious_list)
np.save(os.path.join(data_root_parent, out_iou_heatmap_fname), ious_arr)
plt.imshow(ious_arr)
plt.xlabel("frame j")
plt.ylabel("frame i")
plt.colorbar()
plt.title("pixel-wise IoU heat map")
plt.show()