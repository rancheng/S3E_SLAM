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

import h5py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_root, index):
    data_collections = {}
    torch_data_root = os.path.join(data_root, "torch_data")
    assert os.path.exists(data_root), f"{data_root} does not exist"
    files = glob.glob(os.path.join(data_root, "*.h5"))
    assert len(files) > 0, "No files found"
    h5_name = files[index]
    with h5py.File(h5_name) as f:
        data_collections['ious'] = f['ious'][:]
        data_collections['patch_data'] = f['torch'][:]
        # data_collections['key_points_data'] = f['key_points'][:]
        data_collections['pose'] = f['pose'][:]
    return data_collections

def plot_iou_data(iou_data):
    x_tick = list(range(0, len(iou_data)))
    plt.plot(x_tick, iou_data)

# data, labels = load_data("/mnt/Data/Shared/data_generate/torch_data")