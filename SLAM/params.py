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

import cv2



class Params(object):
    def __init__(self):
        self.feature_detector = cv2.GFTTDetector_create(
            maxCorners=600, minDistance=15.0, 
            qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            bytes=32, use_orientation=False)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 30
        self.matching_distance_ratio = 0.8

        self.virtual_baseline = 0.1  # meters
        self.depth_near = 0.1
        self.depth_far = 10
        self.frustum_near = 0.1 
        self.frustum_far = 50.0
        
        self.pnp_min_measurements = 30
        self.pnp_max_iterations = 10
        self.init_min_points = 30

        self.local_window_size = 10
        self.keyframes_buffer_size = 5
        self.ba_max_iterations = 10

        self.min_tracked_points = 150
        self.min_tracked_points_ratio = 0.75

        self.lc_min_inbetween_keyframes = 15   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 30
        self.lc_inliers_threshold = 13
        self.lc_inliers_ratio = 0.3
        self.lc_distance_threshold = 1.5      # meters
        self.lc_max_iterations = 20
        self.lc_similarity_score_threshold = 0.5  # predicted IoU similarity score
        self.lc_similarity_trusted_threshold = 0.8  # high enough to trust this loop
        self.view_camera_width = 0.05
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000
        self.view_image_width = 400
        self.view_image_height = 250

    def relax_tracking(self, relax=True):
        if relax:
            self.matching_neighborhood = 5
        else:
            self.matching_neighborhood = 2