'''
Iterative Closest Point Implementation
'''

import numpy as np

class ICP():
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros(3)

        X_path = "point_cloud_data/pclX.txt"
        Y_path = "point_cloud_data/pclY.txt"

        self.X = self.parse_point_cloud(X_path)
        self.Y = self.parse_point_cloud(Y_path)

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

    def run(self,):
        corresp = []
        num_corresp = 0

        for x in self.X:
            pass

    def find_closest_point(self, x_point):
        x_trans = self.R @ x_point + self.t
        

