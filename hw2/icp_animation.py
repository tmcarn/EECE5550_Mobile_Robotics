import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

class ICPAnimator:
    def __init__(self, X, Y, R_history, t_history, corresp_history):
        """
        X: source point cloud (N, 3)
        Y: target point cloud (M, 3)
        transformations: list of (R, t) tuples from each ICP iteration
        """
        self.X = X
        self.Y = Y
        self.R_history = R_history
        self.t_history = t_history
        self.corresp_history = corresp_history
        

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def init_plot(self):
        self.ax.clear()
        
        # Plot target points (fixed)
        self.ax.scatter(self.Y[:, 0], self.Y[:, 1], self.Y[:, 2],
                       c='red', marker='.', s=1, alpha=0.3, label='Target')
        
        # Plot initial source points
        self.ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2],
                       c='blue', marker='.', s=1, alpha=0.6, label='Source')
        
        # corresp_idx = np.array(self.corresp_history[0])

        # x_corresp_idx = corresp_idx[:,0]
        # y_corresp_idx = corresp_idx[:,1]

        # x_corresp_points = self.X[x_corresp_idx]
        # y_corresp_points = self.Y[y_corresp_idx]
        
        # for i in range(y_corresp_points.shape[0]):
        #     self.ax.plot([x_corresp_points[i, 0], y_corresp_points[i, 0]],
        #             [x_corresp_points[i, 1], y_corresp_points[i, 1]],
        #             [x_corresp_points[i, 2], y_corresp_points[i, 2]],
        #             'g-', linewidth=0.5, alpha=0.5)
        
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
                       c='red', marker='.', s=1, alpha=0.3, label='Target')
        
        
        R = self.R_history[frame]
        t = self.t_history[frame]
        X_transformed = (R @ self.X.T).T + t
        
        # Plot transformed source
        self.ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],
                       c='blue', marker='.', s=1, alpha=0.6, label='Source')
        
        # Plot correspondences
        corresp_idx = np.array(self.corresp_history[frame-1])

        x_corresp_idx = corresp_idx[:,0]
        y_corresp_idx = corresp_idx[:,1]

        x_corresp_points = X_transformed[x_corresp_idx]
        y_corresp_points = self.Y[y_corresp_idx]
        
        for i in range(y_corresp_points.shape[0]):
            self.ax.plot([x_corresp_points[i, 0], y_corresp_points[i, 0]],
                    [x_corresp_points[i, 1], y_corresp_points[i, 1]],
                    [x_corresp_points[i, 2], y_corresp_points[i, 2]],
                    'g-', linewidth=0.5, alpha=0.5)
        
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