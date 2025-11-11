import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

class VslamAnimator():
    def __init__(self, frames_before_opt=1):
        self.frames_before_opt = frames_before_opt

        self.fig = plt.figure(figsize=(10,5))
        self.ax1 = self.fig.add_subplot(121, projection="3d")
        self.ax2 = self.fig.add_subplot(122)

        # Get Image Paths
        img_dir = "vslam/*.jpg"
        self.img_paths = sorted(glob(img_dir), key=lambda x: int(x.split("_")[1].split(".")[0])) # Sort files based on frame number

        # Load in Pose Information
        with open('pose_history/vslam_pose_history_1.pkl', 'rb') as f:
            self.pose_history = pickle.load(f)

    def update(self, frame_idx):

        self.ax1.clear()
        self.ax2.clear()

        frame = self.pose_history[frame_idx]

        camera_poses = frame["camera_poses"]
        camera_positions = camera_poses[:,:3,3]
        
        tag_poses = frame["tag_poses"]
        tag_positions = tag_poses[:,:3,3]

        # Plot Camera Positions
        self.ax1.plot(camera_positions[:,0],camera_positions[:,1],camera_positions[:,2], linewidth=5, label="Camera Position")

        # Plot Target Positions
        self.ax1.scatter(tag_positions[:,0],tag_positions[:,1],tag_positions[:,2], c="tab:orange", label="April Tags")
        
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()
        self.ax1.set_title(f'VSLAM Camera Trajectory (frame: #{frame_idx})')
        
        # Keep consistent axis limits
        self.ax1.set_xlim(-0.02, 0.1)
        self.ax1.set_ylim(-0.02, 0.1)
        self.ax1.set_zlim(-0.1, 0)

        self.ax1.view_init(elev=-35, azim=105, roll=180)
        
        # Update Image
        img_path = self.img_paths[frame_idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.ax2.imshow(img, cmap="gray")
        self.ax2.axis("off")
        self.ax2.set_title(f"Observed Image (frame: #{frame_idx})")


    def animate(self):
        anim = FuncAnimation(self.fig, 
                             self.update, 
                             frames=len(self.img_paths),
                             interval=1000/30, 
                             repeat=True,
                             blit=False)
        
        # Save as MP4
        anim.save("plots/vslam.mp4", writer="ffmpeg", fps=30, bitrate=1800)

        plt.show()
