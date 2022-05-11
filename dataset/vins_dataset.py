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

import itertools

import torch
import h5py
import numpy as np
from tqdm import tqdm
import os
import glob
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader

# dataset for the embedding training
class VinsDataset(Dataset):
    def __init__(
            self,
            data_root, patch_size="16", split='train'):
        """
        dataset initialization function, generate the iou heatmap, pair index for training
        :param data_root: dataset root, which contains the torch_data folder
        :param patch_size: torch patch size, there are three options: ["16", "32", "64"]
        """
        self._data_root = data_root
        self._patch_size = patch_size
        self._patch_key = "patch_{}".format(self._patch_size)
        assert self._patch_key in ['patch_16', 'patch_32', 'patch_64']
        self._torch_data_root = os.path.join(data_root, "torch_data")
        self._pose_data_root = os.path.join(data_root, "pose")
        assert os.path.exists(self._torch_data_root), f"{self._torch_data_root} does not exist"
        self._torch_data_files = glob.glob(os.path.join(self._torch_data_root, "*.h5"))
        self._torch_data_files = sorted(self._torch_data_files)
        self._dataset_size = len(self._torch_data_files)
        self._split = split
        # iou heatmap
        iou_heatmap_fname = os.path.join(self._data_root, "iou_heatmap.npy")
        if os.path.isfile(iou_heatmap_fname):
            # load the query map of iou data
            self._iou_map = np.load(iou_heatmap_fname)
        else:
            # generate new heatmap
            iou_flist = self._torch_data_files
            ious_list = []
            print("no iou heatmap found, generating new heatmap now")
            for i in tqdm(range(len(iou_flist))):
                h5_name = iou_flist[i]
                with h5py.File(h5_name) as f:
                    ious_list.append(f['ious'][:])
            self._iou_map = np.array(ious_list)
            np.save(iou_heatmap_fname, self._iou_map)
        # balancing the dataset
        balanced_index_sample_fname = os.path.join(self._data_root, "{}_dataset_idx_pairs.npy".format(self._split))
        if os.path.isfile(balanced_index_sample_fname):
            self._idx_pair_list = np.load(balanced_index_sample_fname)
        else:
            # balanced index pair for sample
            self._idx_pair_list = self.data_balancing(self._iou_map)
            # split the dataset to train and val
            train_idx_pair_list, val_idx_pair_list = train_test_split(self._idx_pair_list, shuffle=True)
            print("no idx pairs found, generating one")
            np.save(os.path.join(self._data_root, "{}_dataset_idx_pairs.npy".format("train")), train_idx_pair_list)
            np.save(os.path.join(self._data_root, "{}_dataset_idx_pairs.npy".format("val")), val_idx_pair_list)

    def data_balancing(self, data_arr):
        """
        balance the dataset according to the data_arr distribution
        :param data_arr: input 2D dataset
        :return: list of the pair of index
        """
        # add (-0.1, 0.001] to calculate the number of 0s
        hist = np.histogram(data_arr, bins=[-0.1, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1],
                            range=(-0.1, 1.1))
        digitized_heatmap = np.digitize(data_arr, hist[1])
        idx_x, idx_y = np.where(digitized_heatmap > -1)
        input_x = np.vstack((idx_x, idx_y)).T
        input_y = digitized_heatmap.reshape(-1, 1)
        oversample = SMOTE()
        X, y = oversample.fit_resample(input_x, input_y)
        return X

    def __len__(self):
        return len(self._idx_pair_list)

    def __getitem__(self, i):
        list_pair_i = self._idx_pair_list[i]
        data_collections = {}
        h5_torch_data_name_ref = self._torch_data_files[list_pair_i[0]]
        h5_torch_data_name_target = self._torch_data_files[list_pair_i[1]]
        data_collections['iou_data'] = self._iou_map[list_pair_i[0], list_pair_i[1]]
        with h5py.File(h5_torch_data_name_ref) as f:
            data_collections['idx_ref'] = list_pair_i[0]
            data_collections['patch_data_ref'] = f[self._patch_key][0, :]
            data_collections['key_points_uv_data_ref'] = f['key_points_uv'][0, :]
            data_collections['key_points_xyz_data_ref'] = f['key_points_xyz'][0, :]
            data_collections['pose_ref'] = f['pose'][0, :]
        with h5py.File(h5_torch_data_name_target) as f:
            data_collections['idx_next'] = list_pair_i[1]
            data_collections['patch_data_next'] = f[self._patch_key][0, :]
            data_collections['key_points_uv_data_next'] = f['key_points_uv'][0, :]
            data_collections['key_points_xyz_data_next'] = f['key_points_xyz'][0, :]
            data_collections['pose_next'] = f['pose'][0, :]
        return data_collections

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    data_root = "/mnt/Data/Datasets/S3E_SLAM/data_generate_1"  # "/mnt/Data/Shared/ICRA_DATA_GENERATE/data_generate_1" # "/mnt/Data/Shared/data_generate_2"
    patch_size = 64
    patch_idx = 12
    dataset = VinsDataset(data_root, str(patch_size))
    data = dataset[30]
    im_ref = data['patch_data_ref'][patch_idx, :, :, :3].astype(np.uint8)
    im_next = data['patch_data_next'][patch_idx, :, :, :3].astype(np.uint8)
    im_ref = im_ref[:, :, ::-1]
    im_next = im_next[:, :, ::-1]

    ref_idx = data['idx_ref']
    next_idx = data['idx_next']

    ref_uv = data['key_points_uv_data_ref']
    next_uv = data['key_points_uv_data_next']

    fullsize_im_ref_fname = os.path.join(data_root, "pose_graph", "color_img", "{}_image.png".format(ref_idx))
    fullsize_im_next_fname = os.path.join(data_root, "pose_graph", "color_img", "{}_image.png".format(next_idx))


    # ax1 = plt.subplot(221)
    # plt.imshow(im_ref)
    # #ax1.set_axis_off()
    # ax2 = plt.subplot(222)
    # plt.imshow(im_next)
    # ax3 = plt.subplot(223)
    # fullsize_im_ref = plt.imread(fullsize_im_ref_fname)
    # plt.imshow(fullsize_im_ref)
    # ax3.add_patch(
    #     patches.Rectangle(
    #         xy=(ref_uv[patch_idx, 0] - patch_size/2, ref_uv[patch_idx, 1] - patch_size/2),  # point of origin.
    #         width=patch_size,
    #         height=patch_size,
    #         linewidth=2,
    #         color='#00FF00',
    #         fill=False
    #     )
    # )
    # ax4 = plt.subplot(224)
    # fullsize_im_next = plt.imread(fullsize_im_next_fname)
    # plt.imshow(fullsize_im_next)
    # ax4.add_patch(
    #     patches.Rectangle(
    #         xy=(next_uv[patch_idx, 0] - patch_size/2, next_uv[patch_idx, 1] - patch_size/2),  # point of origin.
    #         width=patch_size,
    #         height=patch_size,
    #         linewidth=2,
    #         color='#00FF00',
    #         fill=False
    #     )
    # )
    fig1 = plt.figure()
    ax1 = plt.subplot(122)
    plt.imshow(im_ref)
    #ax1.set_axis_off()
    ax1.set_xlabel("local patch")
    ax2 = plt.subplot(121)
    plt.imshow(im_next)
    fullsize_im_ref = plt.imread(fullsize_im_ref_fname)
    plt.imshow(fullsize_im_ref)
    ax2.add_patch(
        patches.Rectangle(
            xy=(ref_uv[patch_idx, 0] - patch_size/2, ref_uv[patch_idx, 1] - patch_size/2),  # point of origin.
            width=patch_size,
            height=patch_size,
            linewidth=2,
            color='#00FF00',
            fill=False
        )
    )
    ax2.set_xlabel("full-res image")

    fig2 = plt.figure()
    ax3 = plt.subplot(111)
    plt.imshow(dataset._iou_map)
    ax3.add_patch(
        patches.Rectangle(
            xy=(1805 - 6, 2035 - 6),  # point of origin.
            width=12,
            height=12,
            linewidth=3,
            color="red",
            fill=False
        )
    )
    plt.xlabel("frame i")
    plt.ylabel("frame j")
    plt.colorbar()

    plt.show()
