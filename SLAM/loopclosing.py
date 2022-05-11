import numpy as np
import cv2

import g2o
from g2o.contrib import SmoothEstimatePropagator

import time
from threading import Thread, Lock
from queue import Queue

from collections import defaultdict, namedtuple

from SLAM.optimization import PoseGraphOptimization
from SLAM.components import Measurement

from module.ViT import TwoFViT
from glob import glob
import os
import torch
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# a very simple implementation
class LoopDetection(object):
    def __init__(self, params):
        self.params = params
        self.nns = NearestNeighbors()

    def add_keyframe(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        self.nns.add_item(embedding, keyframe)

    def detect(self, keyframe):
        embedding = keyframe.feature.descriptors.mean(axis=0)
        kfs, ds = self.nns.search(embedding, k=20)

        if len(kfs) > 0 and kfs[0] == keyframe:
            kfs, ds = kfs[1:], ds[1:]
        if len(kfs) == 0:
            return None

        min_d = np.min(ds)
        for kf, d in zip(kfs, ds):
            if abs(kf.id - keyframe.id) < self.params.lc_min_inbetween_keyframes:
                continue
            if (np.linalg.norm(kf.position - keyframe.position) > 
                self.params.lc_max_inbetween_distance):
                break
            if d > self.params.lc_embedding_distance or d > min_d * 1.5:
                break
            return kf
        return None


class LoopDetectionS3E(object):
    def __init__(self, system, params, model_cfg, data_cfg):
        self.params = params
        self.system = system
        self.model_cfg = model_cfg
        self.kf_container = []
        self.kf_dict = {}
        ckpt_out_path = data_cfg['ckpt_output']
        checkpoint_fnames = sorted(list(glob(os.path.join(ckpt_out_path, "*.pth"))))
        if len(checkpoint_fnames) > 0:
            latest_ckpt_path = checkpoint_fnames[-1]
            model_dict = torch.load(latest_ckpt_path)
            intermediate_channels = list(model_cfg['intermediate_channels'])
            num_patches = int(model_cfg['num_patches'])
            patch_size = int(model_cfg['patch_size'])
            pos_dim = int(model_cfg['pos_dim'])
            emb_dim = int(model_cfg['emb_dim'])
            code_dim = int(model_cfg['code_dim'])
            depth = int(model_cfg['depth'])
            heads = int(model_cfg['heads'])
            mlp_dim = int(model_cfg['mlp_dim'])
            pool = model_cfg['pool']
            channels = int(model_cfg['channels'])
            dim_head = int(model_cfg['dim_head'])
            dropout = float(model_cfg['dropout'])
            emb_dropout = float(model_cfg['emb_dropout'])
            # device of model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TwoFViT(
                num_patches=num_patches,
                patch_size=patch_size,
                pos_dim=pos_dim,
                emb_dim=emb_dim,
                code_dim=code_dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                pool=pool,
                channels=channels,  # rgbd
                dim_head=dim_head,
                dropout=dropout,
                emb_dropout=emb_dropout
            )
            self.model.load_state_dict(model_dict["state_dict"])
            self.model.to(self.device)
        else:
            raise ValueError("No Checkpoint found for query network")

    def patchify_keyframe(self, keyframe):
        # create the patches
        kpt_meas = list(keyframe.meas.items())
        num_patches = int(self.model_cfg['num_patches'])
        patch_size = int(self.model_cfg['patch_size'])
        channels = int(self.model_cfg['channels'])
        rdata = torch.zeros((1, num_patches, patch_size, num_patches), dtype=torch.float32)
        image_height, image_width = pair(keyframe.image.shape)
        patch_height, patch_width = pair((patch_size, patch_size))
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        rgb_data = keyframe.image
        depth_data = keyframe.depth
        depth_data = depth_data.reshape(depth_data.shape[0], depth_data.shape[1], 1)
        rgbd_data = np.concatenate((rgb_data, depth_data), axis=-1)
        patcher = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        patched_img = patcher(rgbd_data)
        pos_data = torch.zeros((1, patch_size, 3), dtype=torch.float32)
        for fidx in range(min(len(kpt_meas), patch_size)):
            m1 = kpt_meas[fidx][0]  # measurement
            mappoing = m1.mappoint.position
            key_point = m1.xy
            pos_data[0, fidx, 0] = mappoing[0]
            pos_data[0, fidx, 1] = mappoing[1]
            pos_data[0, fidx, 2] = mappoing[2]
            x_offset = key_point[0] // patch_height
            y_offset = key_point[1] // patch_width
            offset = x_offset * (image_height // patch_height) + y_offset
            rdata[0, fidx] = patched_img[0, offset, :].reshape(patch_height, patch_width, -1)
        patched_img.to(self.device)
        pos_data.to(self.device)
        return (patched_img, pos_data)

    def add_keyframe(self, keyframe):
        self.kf_container.append(keyframe)
        self.kf_dict[keyframe.idx] = self.patchify_keyframe(keyframe)

    def detect(self, keyframe):
        most_matched_kf = None
        max_match_score = -1
        for ref_kf in self.system.graph.keyframes:
            ref_patched_img, ref_pos_data = self.kf_dict[ref_kf.idx]
            query_patched_img, query_pos_data = self.patchify_keyframe(keyframe)
            pred_match_score = self.model(ref_patched_img, ref_pos_data, query_patched_img, query_pos_data)[0][0]
            if pred_match_score >= self.params.lc_similarity_score_threshold and max_match_score < pred_match_score:
                max_match_score = pred_match_score
                most_matched_kf = ref_kf
        return most_matched_kf, max_match_score
        # loop all keyframe's embeddings to find the best match
        # print("Detection Start")
        # for ref_kf in self.system.graph.keyframes:
        #     # get the patches
        #     meas = self.system.graph.keyframes()
        #     list(meas.items())[0][0]
        #     patch_ref = data['patch_data_ref'].float().to(device)
        #     patch_next = data['patch_data_next'].float().to(device)
        #     pos_ref = data['key_points_xyz_data_ref'].float().to(device)
        #     pos_next = data['key_points_xyz_data_next'].float().to(device)


class LoopClosing(object):
    def __init__(self, system, params):
        self.system = system
        self.params = params

        self.loop_detector = LoopDetection(params)
        self.optimizer = PoseGraphOptimization()

        self.loops = []
        self.stopped = False

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def stop(self):
        self.stopped = True
        self._queue.put(None)
        self.maintenance_thread.join()
        print('loop closing stopped')

    def add_keyframe(self, keyframe):
        self._queue.put(keyframe)
        self.loop_detector.add_keyframe(keyframe)

    def add_keyframes(self, keyframes):
        for kf in keyframes:
            self.add_keyframe(kf)

    def maintenance(self):
        last_query_keyframe = None
        while not self.stopped:
            keyframe = self._queue.get()
            if keyframe is None or self.stopped:
                return

            # check if this keyframe share many mappoints with a loop keyframe
            covisible = sorted(
                keyframe.covisibility_keyframes().items(), 
                key=lambda _:_[1], reverse=True)
            if any([(keyframe.id - _[0].id) > 5 for _ in covisible[:2]]):
                continue

            if (last_query_keyframe is not None and 
                abs(last_query_keyframe.id - keyframe.id) < 3):
                continue

            candidate = self.loop_detector.detect(keyframe)
            if candidate is None:
                continue

            query_keyframe = keyframe
            match_keyframe = candidate

            result = match_and_estimate(
                query_keyframe, match_keyframe, self.params)

            if result is None:
                continue
            if (result.n_inliers < max(self.params.lc_inliers_threshold, 
                result.n_matches * self.params.lc_inliers_ratio)):
                continue

            if (np.abs(result.correction.translation()).max() > 
                self.params.lc_distance_threshold):
                continue

            self.loops.append(
                (match_keyframe, query_keyframe, result.constraint))
            query_keyframe.set_loop(match_keyframe, result.constraint)

            # We have to ensure that the mapping thread is on a safe part of code, 
            # before the selection of KFs to optimize
            safe_window = self.system.mapping.lock_window()   # set
            safe_window.add(self.system.reference)
            for kf in self.system.reference.covisibility_keyframes():
                safe_window.add(kf)

            
            # The safe window established between the Local Mapping must be 
            # inside the considered KFs.
            considered_keyframes = self.system.graph.keyframes()

            self.optimizer.set_data(considered_keyframes, self.loops)

            before_lc = [
                g2o.Isometry3d(kf.orientation, kf.position) for kf in safe_window]

            # Propagate initial estimate through 10% of total keyframes 
            # (or at least 20 keyframes)
            d = max(20, len(considered_keyframes) * 0.1)
            propagator = SmoothEstimatePropagator(self.optimizer, d)
            propagator.propagate(self.optimizer.vertex(match_keyframe.id))

            # self.optimizer.set_verbose(True)
            self.optimizer.optimize(20)
            
            # Exclude KFs that may being use by the local BA.
            self.optimizer.update_poses_and_points(
                considered_keyframes, exclude=safe_window)

            self.system.stop_adding_keyframes()

            # Wait until mapper flushes everything to the map
            self.system.mapping.wait_until_empty_queue()
            while self.system.mapping.is_processing():
                time.sleep(1e-4)

            # Calculating optimization introduced by local mapping while loop was been closed
            for i, kf in enumerate(safe_window):
                after_lc = g2o.Isometry3d(kf.orientation, kf.position)
                corr = before_lc[i].inverse() * after_lc

                vertex = self.optimizer.vertex(kf.id)
                vertex.set_estimate(vertex.estimate() * corr)

            self.system.pause()

            for keyframe in considered_keyframes[::-1]:
                if keyframe in safe_window:
                    reference = keyframe
                    break
            uncorrected = g2o.Isometry3d(
                reference.orientation, 
                reference.position)
            corrected = self.optimizer.vertex(reference.id).estimate()
            T = uncorrected.inverse() * corrected   # close to result.correction

            # We need to wait for the end of the current frame tracking and ensure that we
            # won't interfere with the tracker.
            while self.system.is_tracking():
                time.sleep(1e-4)
            self.system.set_loop_correction(T)

            # Updating keyframes and map points on the lba zone
            self.optimizer.update_poses_and_points(safe_window)

            # keyframes after loop closing
            keyframes = self.system.graph.keyframes()
            if len(keyframes) > len(considered_keyframes):
                self.optimizer.update_poses_and_points(
                    keyframes[len(considered_keyframes) - len(keyframes):], 
                    correction=T)

            for query_meas, match_meas in result.shared_measurements:
                new_query_meas = Measurement(
                    query_meas.type,
                    Measurement.Source.REFIND,
                    query_meas.get_keypoints(),
                    query_meas.get_descriptors())
                self.system.graph.add_measurement(
                    query_keyframe, match_meas.mappoint, new_query_meas)
                
                new_match_meas = Measurement(
                    match_meas.type,
                    Measurement.Source.REFIND,
                    match_meas.get_keypoints(),
                    match_meas.get_descriptors())
                self.system.graph.add_measurement(
                    match_keyframe, query_meas.mappoint, new_match_meas)

            self.system.mapping.free_window()
            self.system.resume_adding_keyframes()
            self.system.unpause()

            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    return
            last_query_keyframe = query_keyframe


class LoopClosingS3E(object):
    def __init__(self, system, params, model_cfg, data_cfg, patch_size=16, max_pts=128, auto_fill=True):
        self.system = system
        self.params = params
        self.patch_size = patch_size
        self.max_pts = max_pts
        self.auto_fill = auto_fill
        self._patch_queue = Queue()
        self.loop_detector_s3e = LoopDetectionS3E(system, params, model_cfg, data_cfg)
        self.loop_detector_nn = LoopDetection(params)
        self.optimizer = PoseGraphOptimization()

        self.loops = []
        self.stopped = False

        self._queue = Queue()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.start()

    def stop(self):
        self.stopped = True
        self._queue.put(None)
        self.maintenance_thread.join()
        print('loop closing stopped')

    def add_keyframe(self, keyframe):
        self._queue.put(keyframe)
        self.loop_detector_nn.add_keyframe(keyframe)
        self.loop_detector_s3e.add_keyframe(keyframe)

    def add_keyframes(self, keyframes):
        for kf in keyframes:
            self.add_keyframe(kf)

    def maintenance(self):
        last_query_keyframe = None
        while not self.stopped:
            keyframe = self._queue.get()
            if keyframe is None or self.stopped:
                return

            # check if this keyframe share many mappoints with a loop keyframe
            covisible = sorted(
                keyframe.covisibility_keyframes().items(),
                key=lambda _: _[1], reverse=True)
            if any([(keyframe.id - _[0].id) > 5 for _ in covisible[:2]]):
                continue

            if (last_query_keyframe is not None and
                    abs(last_query_keyframe.id - keyframe.id) < 3):
                continue

            candidate = self.loop_detector_nn.detect(keyframe)
            candidate_s3e, max_score = self.loop_detector_s3e.detect(keyframe)
            if candidate is None and candidate_s3e is None:
                continue
            else:
                if candidate_s3e.idx != candidate.idx and max_score >= self.params.lc_similarity_trusted_threshold:
                    candidate = candidate_s3e  # trust the s3e result

            query_keyframe = keyframe
            match_keyframe = candidate

            result = match_and_estimate(
                query_keyframe, match_keyframe, self.params)

            if result is None:
                continue
            if (result.n_inliers < max(self.params.lc_inliers_threshold,
                                       result.n_matches * self.params.lc_inliers_ratio)):
                continue

            if (np.abs(result.correction.translation()).max() >
                    self.params.lc_distance_threshold):
                continue

            self.loops.append(
                (match_keyframe, query_keyframe, result.constraint))
            query_keyframe.set_loop(match_keyframe, result.constraint)

            # We have to ensure that the mapping thread is on a safe part of code,
            # before the selection of KFs to optimize
            safe_window = self.system.mapping.lock_window()  # set
            safe_window.add(self.system.reference)
            for kf in self.system.reference.covisibility_keyframes():
                safe_window.add(kf)

            # The safe window established between the Local Mapping must be
            # inside the considered KFs.
            considered_keyframes = self.system.graph.keyframes()

            self.optimizer.set_data(considered_keyframes, self.loops)

            before_lc = [
                g2o.Isometry3d(kf.orientation, kf.position) for kf in safe_window]

            # Propagate initial estimate through 10% of total keyframes
            # (or at least 20 keyframes)
            d = max(20, len(considered_keyframes) * 0.1)
            propagator = SmoothEstimatePropagator(self.optimizer, d)
            propagator.propagate(self.optimizer.vertex(match_keyframe.id))

            # self.optimizer.set_verbose(True)
            self.optimizer.optimize(20)

            # Exclude KFs that may being use by the local BA.
            self.optimizer.update_poses_and_points(
                considered_keyframes, exclude=safe_window)

            self.system.stop_adding_keyframes()

            # Wait until mapper flushes everything to the map
            self.system.mapping.wait_until_empty_queue()
            while self.system.mapping.is_processing():
                time.sleep(1e-4)

            # Calculating optimization introduced by local mapping while loop was been closed
            for i, kf in enumerate(safe_window):
                after_lc = g2o.Isometry3d(kf.orientation, kf.position)
                corr = before_lc[i].inverse() * after_lc

                vertex = self.optimizer.vertex(kf.id)
                vertex.set_estimate(vertex.estimate() * corr)

            self.system.pause()

            for keyframe in considered_keyframes[::-1]:
                if keyframe in safe_window:
                    reference = keyframe
                    break
            uncorrected = g2o.Isometry3d(
                reference.orientation,
                reference.position)
            corrected = self.optimizer.vertex(reference.id).estimate()
            T = uncorrected.inverse() * corrected  # close to result.correction

            # We need to wait for the end of the current frame tracking and ensure that we
            # won't interfere with the tracker.
            while self.system.is_tracking():
                time.sleep(1e-4)
            self.system.set_loop_correction(T)

            # Updating keyframes and map points on the lba zone
            self.optimizer.update_poses_and_points(safe_window)

            # keyframes after loop closing
            keyframes = self.system.graph.keyframes()
            if len(keyframes) > len(considered_keyframes):
                self.optimizer.update_poses_and_points(
                    keyframes[len(considered_keyframes) - len(keyframes):],
                    correction=T)

            for query_meas, match_meas in result.shared_measurements:
                new_query_meas = Measurement(
                    query_meas.type,
                    Measurement.Source.REFIND,
                    query_meas.get_keypoints(),
                    query_meas.get_descriptors())
                self.system.graph.add_measurement(
                    query_keyframe, match_meas.mappoint, new_query_meas)

                new_match_meas = Measurement(
                    match_meas.type,
                    Measurement.Source.REFIND,
                    match_meas.get_keypoints(),
                    match_meas.get_descriptors())
                self.system.graph.add_measurement(
                    match_keyframe, query_meas.mappoint, new_match_meas)

            self.system.mapping.free_window()
            self.system.resume_adding_keyframes()
            self.system.unpause()

            while not self._queue.empty():
                keyframe = self._queue.get()
                if keyframe is None:
                    return
            last_query_keyframe = query_keyframe


def depth_to_3d(depth, coords, cam):
    coords = np.array(coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    return np.column_stack([xs, ys, zs])


def match_and_estimate(query_keyframe, match_keyframe, params):
    query = defaultdict(list)
    for kp, desp in zip(
        query_keyframe.feature.keypoints, query_keyframe.feature.descriptors):
        query['kps'].append(kp)
        query['desps'].append(desp)
        query['px'].append(kp.pt)

    match = defaultdict(list)
    for kp, desp in zip(
        match_keyframe.feature.keypoints, match_keyframe.feature.descriptors):
        match['kps'].append(kp)
        match['desps'].append(desp)
        match['px'].append(kp.pt)

    matches = query_keyframe.feature.direct_match(
        query['desps'], match['desps'],
        params.matching_distance, 
        params.matching_distance_ratio)

    query_pts = depth_to_3d(query_keyframe.depth, query['px'], query_keyframe.cam)
    match_pts = depth_to_3d(match_keyframe.depth, match['px'], match_keyframe.cam)

    if len(matches) < params.lc_inliers_threshold:
        return None

    near = query_keyframe.cam.depth_near
    far = query_keyframe.cam.depth_far
    for (i, j) in matches:
        if (near <= query_pts[i][2] <= far):
            query['pt12'].append(query_pts[i])
            query['px12'].append(query['kps'][i].pt)
            match['px12'].append(match['kps'][j].pt)
        if (near <= match_pts[j][2] <= far):
            query['px21'].append(query['kps'][i].pt)
            match['px21'].append(match['kps'][j].pt)
            match['pt21'].append(match_pts[j])

    if len(query['pt12']) < 6 or len(match['pt21']) < 6:
        return None

    T12, inliers12 = solve_pnp_ransac(
        query['pt12'], match['px12'], match_keyframe.cam.intrinsic)

    T21, inliers21 = solve_pnp_ransac(
        match['pt21'], query['px21'], query_keyframe.cam.intrinsic)

    if T12 is None or T21 is None:
        return None

    delta = T21 * T12
    if (g2o.AngleAxis(delta.rotation()).angle() > 0.06 or
        np.linalg.norm(delta.translation()) > 0.06):          # 3° or 0.06m
        return None

    ms = set()
    qd = dict()
    md = dict()
    for i in inliers12:
        pt1 = (int(query['px12'][i][0]), int(query['px12'][i][1]))
        pt2 = (int(match['px12'][i][0]), int(match['px12'][i][1]))
        ms.add((pt1, pt2))
    for i in inliers21:
        pt1 = (int(query['px21'][i][0]), int(query['px21'][i][1]))
        pt2 = (int(match['px21'][i][0]), int(match['px21'][i][1]))
        ms.add((pt1, pt2))
    for i, (pt1, pt2) in enumerate(ms):
        qd[pt1] = i
        md[pt2] = i

    qd2 = dict()
    md2 = dict()
    for m in query_keyframe.measurements():
        pt = m.get_keypoint(0).pt
        idx = qd.get((int(pt[0]), int(pt[1])), None)
        if idx is not None:
            qd2[idx] = m
    for m in match_keyframe.measurements():
        pt = m.get_keypoint(0).pt
        idx = md.get((int(pt[0]), int(pt[1])), None)
        if idx is not None:
            md2[idx] = m
    shared_measurements = [(qd2[i], md2[i]) for i in (qd2.keys() & md2.keys())]

    n_matches = (len(query['pt12']) + len(match['pt21'])) / 2.
    n_inliers = max(len(inliers12), len(inliers21))
    query_pose = g2o.Isometry3d(
        query_keyframe.orientation, query_keyframe.position)
    match_pose = g2o.Isometry3d(
        match_keyframe.orientation, match_keyframe.position)

    # TODO: combine T12 and T21
    constraint = T12
    estimated_pose = match_pose * constraint
    correction = query_pose.inverse() * estimated_pose

    return namedtuple('MatchEstimateResult',
        ['estimated_pose', 'constraint', 'correction', 'shared_measurements', 
        'n_matches', 'n_inliers'])(
        estimated_pose, constraint, correction, shared_measurements, 
        n_matches, n_inliers)


def solve_pnp_ransac(pts3d, pts, intrinsic_matrix):
    val, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d), np.array(pts), 
            intrinsic_matrix, None, None, None, 
            False, 50, 2.0, 0.99, None)
    if inliers is None or len(inliers) < 5:
        return None, None

    T = g2o.Isometry3d(cv2.Rodrigues(rvec)[0], tvec)
    return T, inliers.ravel()
    

# class KFPatchContainer(object):
#     def __init__(self, patch_size):
#         self.n = 0
#         self.patch_size = patch_size
#         self.items = dict()
#         self.data = []
#
#     def add_item(self, vector, item):
#         assert vector.ndim == 1
#         if self.n >= len(self.data):
#             self.data = np.zeros()

class NearestNeighbors(object):
    def __init__(self, dim=None):
        self.n = 0
        self.dim = dim
        self.items = dict()
        self.data = []
        if dim is not None:
            self.data = np.zeros((1000, dim), dtype='float32')

    def add_item(self, vector, item):
        assert vector.ndim == 1
        if self.n >= len(self.data):
            if self.dim is None:
                self.dim = len(vector)
                self.data = np.zeros((1000, self.dim), dtype='float32')
            else:
                self.data.resize(
                    (2 * len(self.data), self.dim) , refcheck=False)
        self.items[self.n] = item
        self.data[self.n] = vector
        self.n += 1

    def search(self, query, k):
        if len(self.data) == 0:
            return [], []

        ds = np.linalg.norm(query[np.newaxis, :] - self.data[:self.n], axis=1)
        ns = np.argsort(ds)[:k]
        return [self.items[n] for n in ns], ds[ns]