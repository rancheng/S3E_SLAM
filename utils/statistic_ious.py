import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# iou heatmap file name
iou_heatmap_fname = "/mnt/Data/Shared/data_generate_2/iou_heatmap.npy"
iou_heatmap = np.load(iou_heatmap_fname)

# add (-0.1, 0.001] to calculate the number of 0s
hist = np.histogram(iou_heatmap,bins=[-0.1,0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1], range=(-0.1, 1.1))
digitized_heatmap = np.digitize(iou_heatmap, hist[1])
idx_x, idx_y = np.where(digitized_heatmap > -1)
input_x = np.vstack((idx_x, idx_y)).T
input_y = digitized_heatmap.reshape(-1, 1)
oversample = SMOTE()
X, y = oversample.fit_resample(input_x, input_y)
# plot the data
fig = plt.figure()
plt.hist(iou_heatmap.reshape(-1, 1), bins=hist[1], rwidth=0.5, label='original distribution')
plt.xlabel('range of data distribution')
plt.ylabel('frequency')
plt.title('histogram of iou heatmap')
plt.legend()

fig2 = plt.figure()
plt.hist(y, bins=[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], rwidth=0.5, label='balanced distribution')
plt.xlabel('range of data distribution')
plt.ylabel('frequency')
plt.title('histogram of iou heatmap')
plt.legend()

fig3 = plt.figure()
ht_y = iou_heatmap[X[:, 0], X[:, 1]]
plt.hist(ht_y, bins=[-0.1,0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1], rwidth=0.5, label='true value distribution')
plt.legend()
plt.show()