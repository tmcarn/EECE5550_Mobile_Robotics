import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam.symbol_shorthand import X, Y
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from glob import glob

# Load in Camera Calibration Parameters
params = np.load("calibration_params/camera_calibration.npz")
K = params['mtx']
fx = K[0,0]
fy = K[1,1]
px = K[0,2]
py = K[1,2]

tag_width = 0.01 # meters

# Initialize Factor Graph and Initial Guesses
graph = gtsam.NonlinearFactorGraph()
initial_estimates = gtsam.Values()

# Initialize Noise Models
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)

# Fix tag_0 to world origin
tag_0_constraint = gtsam.noiseModel.Constrained.All(6) # tag_0 fixed to world origin
tag_0_pose = gtsam.Pose3()
prior_factor = gtsam.PriorFactorPose3(Y(0), tag_0_pose, tag_0_constraint)
graph.add(prior_factor)
initial_estimates.insert(Y(0), tag_0_pose)

# Initialize April Tag Detector
family = 'tag36h11'
detector = Detector(families=family)

# Get Image Paths
img_dir = "vslam/*.jpg"
img_paths = sorted(glob(img_dir), key=lambda x: int(x.split("_")[1].split(".")[0])) # Sort files based on frame number


seen_tags = set()
seen_tags.add(0) # Initial Value already assigned

for img_id, img_path in enumerate(img_paths):
    # Load in image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Initial Camera Pose Guess (Identity)
    # TODO: Use pnp_estimation.py to provide a better initial guess
    initial_estimates.insert(X(img_id), gtsam.Pose3())

    # Detect Corners
    detections = detector.detect(img=img, 
                                estimate_tag_pose=True, 
                                camera_params=(fx, fy, px, py),
                                tag_size=tag_width)
    
    for detection in detections:
        tag_id = detection.tag_id

        # Inital Guess for tag_i
        if tag_id not in seen_tags:
            if not initial_estimates.exists(Y(tag_id)):
                initial_estimates.insert(Y(tag_id), gtsam.Pose3())
                seen_tags.add(tag_id)

        # Rotation and Translation of tag_i in camera_frame j
        R_ct = detection.pose_R 
        t_ct = detection.pose_t
        X_ct = np.concatenate([R_ct, t_ct], axis=1)

        observed_X_ct = gtsam.Pose3(X_ct)
        factor = gtsam.BetweenFactorPose3(X(img_id), Y(tag_id), observed_X_ct, measurement_noise)
        graph.add(factor)

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
result = optimizer.optimize()

print(f"Initial Error: {graph.error(initial_estimates)}")
print(f"Final Error  : {graph.error(result)}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_camera_trajectory(result, seen_cameras, seen_tags):
    """
    Plot camera trajectory and tag positions
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions
    camera_positions = []
    for cam_id in sorted(seen_cameras):
        pose = result.atPose3(X(cam_id))
        pos = pose.translation()
        camera_positions.append([pos[0], pos[1], pos[2]])
    
    camera_positions = np.array(camera_positions)
    
    # Plot camera trajectory
    ax.plot(camera_positions[:, 0], 
            camera_positions[:, 1], 
            camera_positions[:, 2],
            c='tab:blue', linewidth=10, label='Camera Trajectory')
    
    # # Plot camera positions as points
    # ax.scatter(camera_positions[:, 0], 
    #            camera_positions[:, 1], 
    #            camera_positions[:, 2],
    #            c='red', marker='o', label='Camera Poses')
    
    # Label some camera positions
    for i, cam_id in enumerate(sorted(seen_cameras)):
        if i % 25 == 0:  # Label every 25th camera
            ax.text(camera_positions[i, 0], 
                    camera_positions[i, 1], 
                    camera_positions[i, 2],
                    f'  C{cam_id}', fontsize=8)
            # Plot camera pose with coordinate frame
            gtsam_plot.plot_pose3(fig.number, result.atPose3(X(cam_id)), axis_length=tag_width*0.5)
    
    # Plot tag positions
    tag_positions = []
    for tag_id in sorted(seen_tags):
        pose = result.atPose3(Y(tag_id))
        pos = pose.translation()
        tag_positions.append([pos[0], pos[1], pos[2]])
        
        gtsam_plot.plot_pose3(fig.number, result.atPose3(Y(tag_id)), axis_length=tag_width)        
        # Label each tag
        ax.text(pos[0], pos[1], pos[2], 
                f'  T{tag_id}', fontsize=10)
    
    tag_positions = np.array(tag_positions)
    
    ax.scatter(tag_positions[:, 0], 
               tag_positions[:, 1], 
               tag_positions[:, 2],
               c='tab:orange', s=100, label='AprilTags')
    
    # Formatting
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Camera Trajectory and AprilTag Positions', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    all_points = np.vstack([camera_positions, tag_positions])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=-35, azim=105, roll=180)
    
    plt.tight_layout()
    plt.savefig('plots/camera_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call after optimization
plot_camera_trajectory(result, np.arange(500), seen_tags)



        

