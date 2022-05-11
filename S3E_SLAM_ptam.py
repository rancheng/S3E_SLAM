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

import time
from itertools import chain
from collections import defaultdict

from SLAM.covisibility import CovisibilityGraph
from SLAM.optimization import BundleAdjustment
from SLAM.mapping import Mapping
from SLAM.mapping import MappingThread
from SLAM.components import Measurement
from SLAM.motion import MotionModel
from SLAM.loopclosing import LoopClosing, LoopClosingS3E



class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')
            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)



class RGBDPTAM(object):
    def __init__(self, params, model_cfg, data_cfg):
        self.params = params

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)
        self.tracker = Tracking(params)
        self.motion_model = MotionModel(params)

        self.loop_closing = LoopClosingS3E(self, params, model_cfg, data_cfg)
        self.loop_correction = None
        
        self.reference = None        # reference keyframe
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.candidates = []         # candidate keyframes
        self.results = []            # tracking results

        self.status = defaultdict(bool)
        
    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.cloudify()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)
        
        try:
            self.reference = self.graph.get_reference_frame(tracked_map)
            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)

            frame.update_pose(pose)
            self.motion_model.update_pose(
                frame.timestamp, pose.position(), pose.orientation())
            # if len() > 40:
            self.candidates.append(frame)
            self.params.relax_tracking(False)
            self.results.append(True)    # tracking succeed
        except:
            self.params.relax_tracking(True)
            self.results.append(False)   # tracking fail

        remedy = False
        if self.results[-2:].count(False) == 2:
            if (len(self.candidates) > 0 and
                self.candidates[-1].idx > self.preceding.idx):
                frame = self.candidates[-1]
                remedy = True
            else:
                print('tracking failed!')

        if remedy or (
            self.results[-1] and self.should_be_keyframe(frame, measurements)):
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe
            print('new keyframe', keyframe.idx)

        self.set_tracking(False)


    def filter_points(self, frame):
        # local_mappoints = self.graph.get_local_map(self.tracked_map)[0]
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(), 
            len(self.preceding.mappoints()),
            len(self.reference.mappoints()))
        
        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False
        n_matches = len(measurements)
        return (n_matches < self.params.min_tracked_points and
            n_matches > self.params.pnp_min_measurements)


    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']





if __name__ == '__main__':
    import cv2
    import g2o

    import os
    import sys
    import argparse
    import yaml
    from threading import Thread
    
    from SLAM.components import Camera
    from SLAM.components import RGBDFrame
    from SLAM.feature import ImageFeature
    from SLAM.params import Params
    from SLAM.dataset import TUMRGBDDataset, ICLNUIMDataset
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, default='ICL',
        help='dataset (TUM/ICL-NUIM)')
    # parser.add_argument('--path', type=str, help='dataset path', 
    #     default='path/to/your/TUM_RGBD/rgbd_dataset_freiburg1_room')
    parser.add_argument('--path', type=str, help='dataset path', 
        default='path/to/your/ICL-NUIM_RGBD/living_room_traj3_frei_png')
    parser.add_argument(
        '--model_cfg', '-dc',
        type=str,
        required=False,
        default='config/vit_config.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    args = parser.parse_args()

    m_configs = yaml.safe_load(open(args.model_cfg, 'r'))
    model_cfg = m_configs['model']
    data_cfg = m_configs['data']

    if 'tum' in args.dataset.lower():
        dataset = TUMRGBDDataset(args.path)
    elif 'icl' in args.dataset.lower():
        dataset = ICLNUIMDataset(args.path)

    params = Params()
    ptam = RGBDPTAM(params, model_cfg, data_cfg)

    if not args.no_viz:
        from viewer import MapViewer
        viewer = MapViewer(ptam, params)

    height, width = dataset.rgb.shape[:2]
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        width, height, dataset.cam.scale, 
        params.virtual_baseline, params.depth_near, params.depth_far,
        params.frustum_near, params.frustum_far)


    durations = []
    for i in range(len(dataset))[:]:
        feature = ImageFeature(dataset.rgb[i], params)
        depth = dataset.depth[i]
        if dataset.timestamps is None:
            timestamp = i / 20.
        else:
            timestamp = dataset.timestamps[i]

        time_start = time.time()  
        feature.extract()

        frame = RGBDFrame(i, g2o.Isometry3d(), feature, depth, cam, timestamp=timestamp)

        if not ptam.is_initialized():
            ptam.initialize(frame)
        else:
            ptam.track(frame)


        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()
        
        if not args.no_viz:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(ptam.graph.keyframes()))
    print('average time', np.mean(durations))

    ptam.stop()
    if not args.no_viz:
        viewer.stop()

    # collect poses
    tf_list = []
    for kf in ptam.graph.keyframes():
        tf_list.append([dataset.timestamps[kf.idx], kf.pose.translation()[0], kf.pose.translation()[1],
                        kf.pose.translation()[2], kf.pose.orientation().x(), kf.pose.orientation().y(),
                        kf.pose.orientation().z(), kf.pose.orientation().w()])

    # add the current frame to the bottom of the list for further interpolation
    if not ptam.graph.keyframes()[-1].idx == ptam.current.idx: # see if they are the same frame
        tf_list.append([dataset.timestamps[ptam.current.idx], ptam.current.pose.translation()[0], ptam.current.pose.translation()[1],
                        ptam.current.pose.translation()[2], ptam.current.pose.orientation().x(), ptam.current.pose.orientation().y(),
                        ptam.current.pose.orientation().z(), ptam.current.pose.orientation().w()])
    tf_arr = np.array(tf_list)
    np.savetxt(os.path.join(args.path, "kf_pose_result_tum.txt"), tf_arr, fmt="%1.6f")