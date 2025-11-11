import cv2
import numpy as np
from pupil_apriltags import Detector
import gtsam
from gtsam.symbol_shorthand import X, Y
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from glob import glob
import pickle
from vslam_animation import VslamAnimator


class VSLAM():
    def __init__(self, frames_before_opt=500, tag_width=0.01):
        # How many frames should be seen and added to factor graph before optimization
        self.frames_before_opt = frames_before_opt

        # Load in Camera Calibration Parameters
        params = np.load("calibration_params/camera_calibration.npz")
        K = params['mtx']
        fx = K[0,0]
        fy = K[1,1]
        px = K[0,2]
        py = K[1,2]

        self.cam_intrins = (fx, fy, px, py)

        self.tag_width = tag_width # meters

        # Initialize April Tag Detector
        family = 'tag36h11'
        self.detector = Detector(families=family)

        # Initialize Factor Graph and Initial Guesses
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.pose_history = []
        self.best_result = None

        # Initialize Noise Models
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)

        # Fix tag_0 to world origin
        tag_0_constraint = gtsam.noiseModel.Constrained.All(6) # tag_0 fixed to world origin
        tag_0_pose = gtsam.Pose3()
        prior_factor = gtsam.PriorFactorPose3(Y(0), tag_0_pose, tag_0_constraint)
        self.graph.add(prior_factor)
        self.initial_estimates.insert(Y(0), tag_0_pose)

        # Get Image Paths
        img_dir = "vslam/*.jpg"
        self.img_paths = sorted(glob(img_dir), key=lambda x: int(x.split("_")[1].split(".")[0])) # Sort files based on frame number
        
        self.seen_tags = set()
        self.seen_tags.add(0) # Initial Value already assigned

    def update_graph(self, img_id, img_path):
        # Load in image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Add Initial Camera Pose Guess (Identity)
        if not self.initial_estimates.exists(X(img_id)):
            self.initial_estimates.insert(X(img_id), gtsam.Pose3())

        # Detect Corners
        detections = self.detector.detect(img=img, 
                                    estimate_tag_pose=True, 
                                    camera_params=self.cam_intrins,
                                    tag_size=self.tag_width)
        
        for detection in detections:
            tag_id = detection.tag_id

            # Add Inital Guess for tag_i
            if tag_id not in self.seen_tags:
                if not self.initial_estimates.exists(Y(tag_id)):
                    self.initial_estimates.insert(Y(tag_id), gtsam.Pose3())
                    self.seen_tags.add(tag_id)

            # Rotation and Translation of tag_i in camera_frame j
            R_ct = detection.pose_R 
            t_ct = detection.pose_t
            X_ct = np.concatenate([R_ct, t_ct], axis=1)

            observed_X_ct = gtsam.Pose3(X_ct)
            factor = gtsam.BetweenFactorPose3(X(img_id), Y(tag_id), observed_X_ct, self.measurement_noise)
            self.graph.add(factor)

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)   
        return optimizer.optimize()

    def save_pose_information(self, result, img_id):
        # Extract poses at this frame
        frame_data = {
            'camera_poses': {},
            'tag_poses': {}
        }

        # Save Updated Camera Poses
        cam_poses = np.zeros((img_id+1,4,4))
        for cam_id in range(img_id + 1):
            cam_pose = result.atPose3(X(cam_id))
            cam_poses[cam_id] = cam_pose.matrix()
        
        frame_data['camera_poses'] = cam_poses
        
        # Save all tag poses seen so far
        tag_poses = np.zeros((24,4,4))
        for tag_id in self.seen_tags:
            tag_pose = result.atPose3(Y(tag_id))
            tag_poses[tag_id] = tag_pose.matrix()
            
        frame_data['tag_poses'] = tag_poses

        return frame_data
    
    def get_error(self, result):
        return self.graph.error(result)
    
    def run(self):
        pose_history = []
        for img_id, img_path in enumerate(self.img_paths):
            self.update_graph(img_id, img_path)
            if (img_id + 1) % self.frames_before_opt == 0:
                self.best_result = self.optimize()
                
                if self.frames_before_opt == 1: # Contents only saved for animation
                    frame_data = self.save_pose_information(self.best_result, img_id)
                    pose_history.append(frame_data)
                print(f'Error after Optimization:{self.get_error(self.best_result)}')

        if self.frames_before_opt == 1: # Contents only saved for animation
            with open(f'pose_history/vslam_pose_history_{self.frames_before_opt}.pkl', 'wb') as f:
                pickle.dump(pose_history, f)

    def plot_final_trajectory(self):
        """
        Plot camera trajectory and tag positions
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract camera positions
        camera_positions = []
        for cam_id in range(len(self.img_paths)):
            pose = self.best_result.atPose3(X(cam_id))
            pos = pose.translation()
            camera_positions.append([pos[0], pos[1], pos[2]])
        
        camera_positions = np.array(camera_positions)
        
        # Plot camera trajectory
        ax.plot(camera_positions[:, 0], 
                camera_positions[:, 1], 
                camera_positions[:, 2],
                c='tab:blue', linewidth=10, label='Camera Trajectory')
        
        # Label some camera positions
        for cam_id in range(len(self.img_paths)):
            if cam_id % 25 == 0:  # Label every 25th camera
                ax.text(camera_positions[cam_id, 0], 
                        camera_positions[cam_id, 1], 
                        camera_positions[cam_id, 2],
                        f'  C{cam_id}', fontsize=10)
                
                # Plot camera pose with coordinate frame
                gtsam_plot.plot_pose3(fig.number, self.best_result.atPose3(X(cam_id)), axis_length=self.tag_width*0.5)
        
        # Plot tag positions
        tag_positions = []
        for tag_id in sorted(self.seen_tags):
            pose = self.best_result.atPose3(Y(tag_id))
            pos = pose.translation()
            tag_positions.append([pos[0], pos[1], pos[2]])
            
            gtsam_plot.plot_pose3(fig.number, self.best_result.atPose3(Y(tag_id)), axis_length=self.tag_width)        
            
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
        
        # Keep consistent axis limits
        ax.set_xlim(-.02, 0.1)
        ax.set_ylim(-.02, 0.1)
        ax.set_zlim(-0.1, 0)
        
        ax.view_init(elev=-35, azim=105, roll=180)
        
        plt.tight_layout()
        plt.savefig('plots/camera_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()


# Optimized on Every Frame (for animation)
# vslam = VSLAM(frames_before_opt=1)
# result = vslam.run()
# vslam.plot_final_trajectory()

# anim = VslamAnimator(vslam.frames_before_opt)
# anim.animate()

# Optimized only at the end (after 500 frames)
vslam = VSLAM()
result = vslam.run()
vslam.plot_final_trajectory()
