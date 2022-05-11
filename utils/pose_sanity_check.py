# Copyright 2021 Ran Cheng
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
from evo.core import metrics
from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)
from evo.core import sync
import copy
import pprint
import numpy as np
from evo.tools import plot
import matplotlib.pyplot as plt


# temporarily override some package settings
from evo.tools.settings import SETTINGS
SETTINGS.plot_usetex = False
SETTINGS.plot_axis_marker_scale = 0.1

from evo.tools import file_interface

# ref_file = "./data/stamped_traj_estimate_mono_pg.txt"
# est_file = "./data/stamped_traj_estimate_mono_vio.txt"

ref_file = "/home/ran/PycharmProjects/s3e_project/data/experments/gen5/stamped_traj_estimate_mono_pg.txt"
vins_optimized_file = "/home/ran/PycharmProjects/s3e_project/data/experments/gen5/orb3_trajectory_tum.txt"
est_file = "/home/ran/PycharmProjects/s3e_project/data/experments/gen5/stamped_traj_estimate_mono_vio.txt"

traj_ref = file_interface.read_tum_trajectory_file(ref_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)
traj_vins_opt = file_interface.read_tum_trajectory_file(vins_optimized_file)

max_diff = 0.01

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
traj_ref2, traj_vins_opt = sync.associate_trajectories(traj_ref, traj_vins_opt, max_diff)
traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
traj_vins_opt.align(traj_ref2, correct_scale=True, correct_only_scale=False)
fig, ax = plt.subplots()
traj_by_label = {
    "VINSMono+S3E-GNN": traj_est,
    "VINSMono": traj_vins_opt,
    "GT": traj_ref
}
print("SETTINGS.plot_axis_marker_scale: {}".format(SETTINGS.plot_axis_marker_scale))
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plot.draw_coordinate_axes(ax, traj_ref, plot.PlotMode.xyz,
                          SETTINGS.plot_axis_marker_scale)
plot.draw_coordinate_axes(ax, traj_est, plot.PlotMode.xyz,
                          SETTINGS.plot_axis_marker_scale)
plt.show()