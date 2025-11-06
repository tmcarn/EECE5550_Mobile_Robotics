import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam.symbol_shorthand import X, L

# Load in Camera Calibration Parameters
params = np.load("calibration_params/camera_calibration.npz")
K = params['mtx']
fx = K[0,0]
fy = K[1,1]
px = K[0,2]
py = K[1,2]

tag_width = 0.01 # meters
spacing = tag_width / 2.0

# Initialize Corner Point Coordinates in Body-Centric Frame [c1, c2, c3, c4]
corners_3d = np.array([[-spacing, spacing, 0],
                        [spacing, spacing, 0],
                        [spacing, -spacing, 0], 
                        [-spacing, -spacing, 0]]
                    )

# Initialize April Tag Detector
family = 'tag36h11'
detector = Detector(families=family)

# Load in Image
img_path = "vslam/frame_0.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Detect Corners
detections = detector.detect(img)

# Find April Tag 0
tag_id = 0
id_idx = None

for i, detection in enumerate(detections):
    if detection.tag_id == tag_id:
        id_idx = i
        break

tag_id = detections[id_idx]
corners_2d = tag_id.corners

# Initialize Camera Calibration
skew = 0.0
cam_cal = gtsam.Cal3_S2(fx=fx, fy=fy, s=skew, u0=px, v0=py)

noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)

G = gtsam.NonlinearFactorGraph()

for i in range(4):
    point_2d = gtsam.Point2(x=corners_2d[i,0], y=corners_2d[i,1])
    point_3d = gtsam.Point3(x=corners_3d[i,0], y=corners_3d[i,1], z=corners_3d[i,2])

    factor = gtsam.GenericProjectionFactorCal3_S2(point_2d, noise, X(0), L(i), cam_cal)
    G.add(factor)








