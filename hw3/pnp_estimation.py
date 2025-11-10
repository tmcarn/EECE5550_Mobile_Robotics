import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam.symbol_shorthand import X, L
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

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
detections = detector.detect(img=img, 
                             estimate_tag_pose=True, 
                             camera_params=(fx,fy, px, py),
                             tag_size=tag_width)

# Find April Tag i
tag_id = 0
id_idx = None

for i, detection in enumerate(detections):
    if detection.tag_id == tag_id:
        id_idx = i
        break

if id_idx == None:
    print(f'AprilTag_{tag_id} not found')
    exit()

tag = detections[id_idx]

corners_2d = tag.corners

# AprilTag initial pose estimate (in camera frame)
R_camera_tag = detection.pose_R
t_camera_tag = detection.pose_t.flatten()

# Invert to get camera pose in tag frame
init_pose_R = R_camera_tag.T
init_pose_t = -init_pose_R @ t_camera_tag

# Initialize Camera Calibration
skew = 0.0
cam_cal = gtsam.Cal3_S2(fx=fx, fy=fy, s=skew, u0=px, v0=py)

# Initialize Noise Models
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 0.1)
landmark_constraint = gtsam.noiseModel.Constrained.All(3) # Landmarks rigidly fixed to world frame

graph = gtsam.NonlinearFactorGraph()
initial_estimates = gtsam.Values()

# Build Up Graph
init_pose = gtsam.Pose3(gtsam.Rot3(init_pose_R), 
                        gtsam.Point3(init_pose_t[0], init_pose_t[1], init_pose_t[2]))


initial_estimates.insert(X(0), init_pose)

for i in range(4):
    point_2d = gtsam.Point2(x=corners_2d[i,0], y=corners_2d[i,1])
    point_3d = gtsam.Point3(x=corners_3d[i,0], y=corners_3d[i,1], z=corners_3d[i,2])

    factor = gtsam.GenericProjectionFactorCal3_S2(point_2d, measurement_noise, X(0), L(i), cam_cal)
    prior_factor = gtsam.PriorFactorPoint3(L(i), point_3d, landmark_constraint)

    initial_estimates.insert(L(i), point_3d)
    graph.add(factor)
    graph.add(prior_factor)
    

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
result = optimizer.optimize()


print(f"Initial Error: {graph.error(initial_estimates)}")
print(f"Final Error  : {graph.error(result)}")


"""
Use GTSAM's built-in plotting functions
"""
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Get optimized camera pose
camera_pose = result.atPose3(X(0))

# Plot camera pose with coordinate frame
gtsam_plot.plot_pose3(fig.number, camera_pose, axis_length=tag_width)

# Draw tag outline
corners = np.array([result.atPoint3(L(i)) for i in range(4)])
corners_closed = np.vstack([corners, corners[0]])
ax.plot(corners_closed[:, 0], corners_closed[:, 1], corners_closed[:, 2],
        'b-', linewidth=3, label='ArUco Tag')

# Add world coordinate frame at origin
origin_pose = gtsam.Pose3()  # Identity pose at origin
gtsam_plot.plot_pose3_on_axes(ax, origin_pose, axis_length=tag_width,
                                P=None, scale=1.0)

# Label Axes
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title(f'GTSAM: Camera Pose Estimation for AprilTag_{tag_id}', fontsize=14, fontweight='bold')


pos = np.array(camera_pose.translation())
rot = camera_pose.rotation().matrix()

# Set axis limits
max_range = max(abs(pos[0]), abs(pos[1]), abs(pos[2])) * 1.2
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, 0])\

ax.text2D(0.95, 0.95, f"Estimated Camera Pose:\n\nR:{np.array2string(rot, precision=3)}\n\nt: {np.array2string(pos, precision=3)}\n\nProjection Error: \n{graph.error(result)}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.view_init(elev=-45, azim=136, roll=180)
plt.tight_layout()
plt.savefig('plots/gtsam_tag_0.png', dpi=300, bbox_inches='tight')
plt.show()
