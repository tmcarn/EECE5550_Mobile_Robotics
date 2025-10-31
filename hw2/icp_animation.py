import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

class ICPAnimator:
    def __init__(self, X, Y, R_history, t_history):
        """
        X: source point cloud (N, 3)
        Y: target point cloud (M, 3)
        transformations: list of (R, t) tuples from each ICP iteration
        """
        self.X = X
        self.Y = Y
        self.R_history = R_history
        self.t_history = t_history
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def init_plot(self):
        self.ax.clear()
        
        # Plot target points (fixed)
        self.ax.scatter(self.Y[:, 0], self.Y[:, 1], self.Y[:, 2],
                       c='tab:red', s=1, alpha=0.5, label='Target')
        
        # Plot initial source points with no transformation
        self.ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2],
                       c='tab:blue', s=1, alpha=0.5, label='Source')
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title('ICP Iteration 0')
        
        # Set consistent axis limits
        all_points = np.vstack([self.X, self.Y])
        margin = 0.1
        self.ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        self.ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        self.ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        
        return self.ax,
    
    def update(self, frame):
        self.ax.clear()
        
        # Plot target (fixed)
        self.ax.scatter(self.Y[:, 0], self.Y[:, 1], self.Y[:, 2],
                       c='tab:red', s=1, alpha=0.5, label='Target')
        
        
        R = self.R_history[frame]
        t = self.t_history[frame]
        X_transformed = (R @ self.X.T).T + t
        
        # Plot transformed source
        self.ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],
                       c='tab:blue', s=1, alpha=0.5, label='Source')
        
        # Display R and t values for each iteration
        self.ax.text2D(0.95, 0.95, f"R: {np.array2string(R, precision=3)}\n\nt: {np.array2string(t, precision=3)}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        self.ax.set_title(f'ICP Iteration {frame}')
        
        # Keep consistent axis limits
        all_points = np.vstack([self.X, self.Y])
        margin = 0.1
        self.ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        self.ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        self.ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        
        return self.ax,

    def plot_total_change(self):
        """Create side-by-side before and after comparison"""
        
        self.ax.clear()

        fig = plt.figure(figsize=(16, 7))
        
        # BEFORE (left subplot)
        self.ax = fig.add_subplot(121, projection='3d')
        self.update(0)
        
        # AFTER (right subplot)
        self.ax = fig.add_subplot(122, projection='3d')
        self.update(len(self.R_history) - 1)  # Use last frame
        
        plt.tight_layout()
        plt.savefig('plots/prob2_before_after.png', dpi=300, bbox_inches='tight')
        plt.show()

    
    def animate(self, interval=500, repeat=True):
        anim = FuncAnimation(self.fig, self.update, 
                           frames=self.R_history.shape[0],
                           init_func=self.init_plot,
                           interval=interval, 
                           repeat=repeat,
                           blit=False)
        
        # Save as MP4
        writer = FFMpegWriter(fps=5, bitrate=1800)
        anim.save("plots/prob2_animation.mp4", writer=writer)

        plt.show()
        return anim