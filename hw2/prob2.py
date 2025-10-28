'''
Iterative Closest Point Implementation
'''

import numpy as np
import math
from scipy.spatial import KDTree
from icp_animation import ICPAnimator
from matplotlib import pyplot as plt


class ICP():
    def __init__(self):
        # Initialize Parameters
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.R_history = [self.R]
        self.t_history = [self.t]
        self.corresp_history = []

        # Set Hyperparameters
        self.max_distance = 0.25
        self.num_iters = 30

        X_path = "/Users/theo/EECE 5550/hw2/point_cloud_data/pclX.txt"
        Y_path = "/Users/theo/EECE 5550/hw2/point_cloud_data/pclY.txt"

        self.X = self.parse_point_cloud(X_path)
        self.Y = self.parse_point_cloud(Y_path)

        self.Y_kdtree = KDTree(self.Y)

    def parse_point_cloud(self, pcl_path):
        '''
        Converts .txt file to np array with shape (n, 3)
        '''
        with open(pcl_path, 'r') as file:
            lines = file.readlines()
            coords = []
            for line in lines:
                coord = line.split(' ')
                coords.append(coord)
        
        coords = np.array(coords, dtype=float)
        return coords

    def estimate_correspondences(self):
        corresp_idx = []

        X_trans = (self.R @ self.X.T).T + self.t

        for x_idx, x in enumerate(X_trans):
            distance, y_idx = self.find_closest_point(x)
            if math.sqrt(distance) < self.max_distance:
                corresp_idx.append([x_idx, y_idx])

        self.corresp_history.append(corresp_idx)
        return np.array(corresp_idx)


    def find_closest_point(self, x_trans):
        '''
        Returns the point in set Y that is closest to input point in set X with current R and t parameters
        '''
        cp_distance, cp_idx = self.Y_kdtree.query(x_trans)
        return cp_distance, cp_idx
    
    def compute_optimal_rigid_registration(self, corresp_idx):
        '''
        Updates self.R and self.t based on input correspondences
        '''
        x_corresp = self.X[corresp_idx[:,0]]
        y_corresp = self.Y[corresp_idx[:,1]]

        x_centroid = np.mean(x_corresp, axis=0)
        y_centroid = np.mean(y_corresp, axis=0)

        x_centroid_dist = x_corresp - x_centroid
        y_centroid_dist = y_corresp - y_centroid

        K = x_corresp.shape[0]
        W = y_centroid_dist.T @ x_centroid_dist / K

        U, S, Vt = np.linalg.svd(W)
        V = Vt.T

        det = np.linalg.det(U @ V)
        diag_vec = np.ones(U.shape[1])
        diag_vec[-1] = det # Handles Reflection Cases
        diag = np.diag(diag_vec)

        self.R = U @ diag @ V.T
        self.t = y_centroid - self.R @ x_centroid

        self.R_history.append(self.R)
        self.t_history.append(self.t)

    def rmse(self):
        X_trans = (self.R @ self.X.T).T + self.t
        difference = self.Y - X_trans
        distance = np.linalg.norm(difference) ** 2
        avg_dist = np.mean(distance)
        rmse = np.sqrt(avg_dist)
        return rmse

    def viz(self):
        self.R_history = np.array(self.R_history)
        self.t_history = np.array(self.t_history)

        animator = ICPAnimator(self.X, self.Y, self.R_history, self.t_history, self.corresp_history)

        animator.animate()

        def plot_rmse():
            fig, ax = plt.subplots()

            ax.plot(self.rmse_history)
            plt.show()

        plot_rmse()



    def run(self):
        for i in range(self.num_iters):
            corresp_idx = self.estimate_correspondences()
            self.compute_optimal_rigid_registration(corresp_idx)
            print(self.rmse())
        
        self.viz()


icp = ICP()
icp.run()