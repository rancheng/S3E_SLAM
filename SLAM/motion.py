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
import g2o



class MotionModel(object):
    def __init__(self, params):
        self.params = params

        self.timestamp = None
        self.position = np.zeros(3)
        self.orientation = g2o.Quaternion()
        self.covariance = None    # pose covariance

        self.v_linear = np.zeros(3)    # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])

        self.initialized = False
        self.damp = 0.95   # damping factor

    def current_pose(self):
        '''
        Get the current camera pose.
        '''
        return (g2o.Isometry3d(self.orientation, self.position), 
            self.covariance)

    def predict_pose(self, timestamp):
        '''
        Predict the next camera pose.
        '''
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), 
                self.covariance)
        
        dt = timestamp - self.timestamp

        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp, 
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)

        position = self.position + self.v_linear * dt * self.damp
        orientation = self.orientation * delta_orientation

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, 
            new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            if np.linalg.norm(self.v_linear) > 0:
                v0 = np.linalg.norm(self.v_linear)
                v1 = np.linalg.norm(v_linear)
                if v1 / max(v0, 1.0) > 5:
                    v_linear = self.v_linear
            self.v_linear = v_linear

            delta_q = self.orientation.inverse() * new_orientation
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt
            
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation
        self.covariance = new_covariance
        self.initialized = True

    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        '''
        Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)
        '''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)

        current = g2o.Isometry3d(self.orientation, self.position)
        current = current * correction

        self.position = current.position()
        self.orientation = current.orientation()

        self.v_linear = (
            correction.inverse().orientation() * self.v_linear)
        self.v_angular_axis = (
            correction.inverse().orientation() * self.v_angular_axis)