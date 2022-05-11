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
import argparse

def vins2TUM(fname):
   pose_data = np.genfromtxt(fname, dtype=float, delimiter=",")
   pose_data[:, 0] /= 1e9
   pose_data = pose_data[:, :8]
   out_fname = fname.replace(".csv", "_tum.txt")
   np.savetxt(out_fname, pose_data)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--result_fname", type=str,
                       help="ORB-SLAM3 output result file name, please input absolute path")
   args = parser.parse_args()
   vins2TUM(args.result_fname)