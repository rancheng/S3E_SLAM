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
from evo.core import metrics
from evo.tools import log

log.configure_logging(verbose=True, debug=True, silent=False)
from evo.core import sync
import copy
import pprint
import numpy as np
from evo.tools import plot
import matplotlib.pyplot as plt
import argparse

# temporarily override some package settings
from evo.tools.settings import SETTINGS

SETTINGS.plot_usetex = False
SETTINGS.plot_axis_marker_scale = 0.1

from evo.tools import file_interface


def vis_traj(traj_fname):
    ref_file = traj_fname  # "/home/ran/Documents/D435i_pose_graph/pose_graph/vins_result_loop_tum.txt"

    traj_ref = file_interface.read_tum_trajectory_file(ref_file)

    max_diff = 0.01

    fig, ax = plt.subplots()
    traj_by_label = {
        "vins traj": traj_ref
    }
    # print("SETTINGS.plot_axis_marker_scale: {}".format(SETTINGS.plot_axis_marker_scale))
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    plot.draw_coordinate_axes(ax, traj_ref, plot.PlotMode.xyz,
                              SETTINGS.plot_axis_marker_scale)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_fname", type=str,
                        default="/home/ran/Documents/D435i_pose_graph/pose_graph/vins_result_loop_tum.txt",
                        help="Trajectory file name (TUM format), please input absolute path")
    args = parser.parse_args()
    vis_traj(args.traj_fname)
