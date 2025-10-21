import numpy as np
# from matplotlib import pyplot as plt

class EKFSim():
    def __init__(self):
        self.t = 0

        # Initial States
        self.true_pos = np.array([[0,0]]).T
        self.landmarks = np.array([[5,5],[-5, 5]]) #[(l_1x,l_1y), l_2x, l2_y)]

        self.estim_pos = np.array([[0,0]]).T # Column Vector for Position        

        # Covariance Matricies
        self.sigma = np.eye(2)   # Final Positional Uncertainty
        self.R = 0.1 * np.eye(2) # Process Uncertainty
        self.Q = 0.5 * np.eye(2) # Measurement Uncertainty

    def update_true_pos(self, t):
        process_noise = np.random.multivariate_normal(np.zeros(2), self.R).reshape(-1, 1)
        self.true_pos = self.true_pos + self.get_vel(t) + process_noise

    
    def measurment_model(self, pos, noisy:bool):
        '''
        Input: [x,y] position
        Output: [d1, d1] (distance to landmark_1 and landmark_2)

        L2 Distance (optionally) plus sensor noise delta ~ N (0, Q)
        '''
        pos_mat = np.concatenate([pos.T, pos.T], axis=0)
        difference_vectors = pos_mat - self.landmarks
        measurement = np.linalg.norm(difference_vectors, axis=1, keepdims=True)

        if noisy:
            measurement_noise = np.array([np.random.multivariate_normal(np.zeros(2), self.Q)]).T
            return measurement + measurement_noise
        else:
            return measurement
        
    
    def get_measurement_jacobian(self, pos):
        pos_mat = np.concatenate([pos.T, pos.T], axis=0)
        difference_vectors = pos_mat - self.landmarks
        distances = self.measurment_model(pos, noisy=False)
        H = difference_vectors / distances  # Broadcasting: (2,2) / (2,1) = (2,2)
        return H

    
    def state_propagation_step(self, pos, t):
        dead_rek_pos = pos + self.get_vel(t)
        dead_rek_sigma = self.sigma + self.R
        return dead_rek_pos, dead_rek_sigma


    def correction_step(self, dead_rek_pos, dead_rek_sigma):
        H = self.get_measurement_jacobian(dead_rek_pos)
        k_gain = dead_rek_sigma @ H.T @ np.linalg.inv(H @ dead_rek_sigma @ H.T + self.Q)
        # Calculate Innovation (difference between true measurement and expected measurement)
        true_measurement = self.measurment_model(self.true_pos, noisy=True)
        pred_measurement = self.measurment_model(dead_rek_pos, noisy=False)
        innovation = (true_measurement - pred_measurement)
        # Estimate Position based on Kalman Gain and Innovation
        self.estim_pos = dead_rek_pos + k_gain @ (innovation)
        # Calculate Certainty
        self.sigma = (np.eye(2) - k_gain @ H) @ dead_rek_sigma


    def get_vel(self, t):
        '''
        Defines a stepwise function for velocity
        '''
        if 0<=t<=10:
            return np.array([[1,0]]).T
        elif 10<t<=20:
            return np.array([[0,-1]]).T
        elif 20<t<=30:
            return np.array([[-1,0]]).T
        elif 30<t<=40:
            return np.array([[0,1]]).T
        else:
            print("'t' is out of bounds")
            raise Exception
        
    def run(self):
        for t in range(1, 41):
            # Update True Position
            self.update_true_pos(t)
            # State Propagation
            dead_rek_pos, dead_rek_sigma = self.state_propagation_step(self.estim_pos, t)
            # Correction Step
            self.correction_step(dead_rek_pos,dead_rek_sigma)
            print(f"Estimated Position:{self.estim_pos}")
            print(f"True Position: {self.true_pos}")
            

            

# Unit Tests:
sim = EKFSim()
sim.run()
