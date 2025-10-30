import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse

class EKFSim():
    def __init__(self):
        self.t = 0
        self.max_t = 40
        self.dt = 0.5

        # Initial States
        self.true_pos = np.array([0,0]) # shape: (2,)
        self.landmarks = np.array([[-5,5],[5, 5]]) # [[l_1x,l_1y], [l_2x, l2_y]]

        self.estim_pos = np.array([0,0]) # shape: (2,)        

        # Covariance Matricies
        self.estim_sigma = np.eye(2) # Estimated Position Uncertainty
        self.R = 0.1 * np.eye(2) # Process Uncertainty
        self.Q = 0.5 * np.eye(2) # Measurement Uncertainty

        self.true_pos_history = []
        self.estim_pos_history = []
        self.estim_sigma_history = []

    def update_true_pos(self, t):
        '''
        This function keeps track of the true position of the robot (unknown to the actual robot)

        Args:
            t (int): current timestep

        Returns:
            self.true_pos (2,): updated true position
        '''
        process_noise = np.random.multivariate_normal(np.zeros(2), self.R)
        self.true_pos = self.true_pos + (self.get_vel(t) * self.dt) + process_noise

    
    def measurment_model(self, pos, noisy:bool):
        '''
        Input: [x,y] position
        Output: [d1, d1] (distance to landmark_1 and landmark_2)

        L2 Distance (optionally) plus sensor noise delta ~ N (0, Q)
        '''
        # pos_mat = np.concatenate([pos.T, pos.T], axis=0)
        difference_vectors = pos - self.landmarks
        measurement = np.linalg.norm(difference_vectors, axis=1)

        if noisy:
            measurement_noise = np.random.multivariate_normal(np.zeros(2), self.Q)
            return measurement + measurement_noise
        
        return measurement
        
    
    def get_measurement_jacobian(self, pos):
        # pos_mat = np.concatenate([pos.T, pos.T], axis=0)
        difference_vectors = pos - self.landmarks
        distances = self.measurment_model(pos, noisy=False)
        H = difference_vectors / distances[:, np.newaxis]
        return H

    
    def state_propagation_step(self, pos, t):
        dead_rek_pos = pos + (self.get_vel(t) * self.dt)
        dead_rek_sigma = self.estim_sigma + self.R
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
        self.estim_sigma = (np.eye(2) - k_gain @ H) @ dead_rek_sigma


    def get_vel(self, t):
        '''
        Defines a stepwise function for velocity
        '''
        if 0<=t<=10:
            return np.array([1,0])
        elif 10<t<=20:
            return np.array([0,-1])
        elif 20<t<=30:
            return np.array([-1,0])
        elif 30<t<=40:
            return np.array([0,1])
        else:
            print(f"t={t} is out of bounds")
            raise Exception
        
    def run(self):
        for i in range(int(self.max_t/self.dt)):
            self.t += 0.5 # Update current time

            # Update True Position
            self.update_true_pos(self.t)
            # State Propagation
            dead_rek_pos, dead_rek_sigma = self.state_propagation_step(self.estim_pos, self.t)
            # Correction Step
            self.correction_step(dead_rek_pos,dead_rek_sigma)

            self.estim_pos_history.append(self.estim_pos)
            self.estim_sigma_history.append(self.estim_sigma)
            self.true_pos_history.append(self.true_pos)

        self.estim_pos_history = np.array(self.estim_pos_history).squeeze()
        self.estim_sigma_history = np.array(self.estim_sigma_history).squeeze()
        self.true_pos_history = np.array(self.true_pos_history).squeeze()

    def viz(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        l1 = Circle(self.landmarks[0], radius=1.0, facecolor="tab:green")
        ax.add_patch(l1)
        ax.text(self.landmarks[0,0],  # Label Point with time step
                    self.landmarks[0,1], 
                    f"L_1",
                    fontsize=5,
                    fontweight='bold', 
                    color='white',
                    ha='center',        
                    va='center'
                    )
        
        l2 = Circle(self.landmarks[1], radius=1.0, facecolor="tab:green")
        ax.add_patch(l2)
        ax.text(self.landmarks[1,0],  # Label Point with time step
                    self.landmarks[1,1], 
                    f"L_2",
                    fontsize=5,
                    fontweight='bold', 
                    color='white',
                    ha='center',        
                    va='center'
                    )

        def get_cov_ellipse(cov, n_std=3):
                eig_vals, eig_vecs = np.linalg.eig(cov)

                # Sort Eig Vals/Vectors
                order = np.argsort(eig_vals)[::-1] # Decending Order (major axis first)
                eig_vals = eig_vals[order]
                eig_vecs = eig_vecs[:, order]

                major = 2 * n_std * np.sqrt(eig_vals[0])
                minor = 2 * n_std * np.sqrt(eig_vals[1])
                angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

                return major, minor, angle

        for i in range(0, self.true_pos_history.shape[0], 8): # Intervals of t=4
            true_pos = self.true_pos_history[i]
            estim_pos = self.estim_pos_history[i]
            estim_sigma = self.estim_sigma_history[i]
            
            major, minor, angle = get_cov_ellipse(estim_sigma)

            true_pos_plot = Circle(true_pos, radius=0.5, facecolor="purple")
            estim_certainty_plot = Ellipse(estim_pos, width=major, height=minor, angle=angle, facecolor="orange", alpha=0.2)
            
            # ax.add_patch(true_pos_plot)
            ax.scatter(true_pos[0], true_pos[1], c="tab:blue", s=200)
            ax.text(true_pos[0],  # Label Point with time step
                    true_pos[1], 
                    f"t={int(i/2)}",
                    fontsize=5,
                    fontweight='bold', 
                    color='white',
                    ha='center',        
                    va='center'
                    )
            
            ax.scatter(estim_pos[0], estim_pos[1], c="orange")
            ax.add_patch(estim_certainty_plot)
            

        # Must set axis limits when using patches
        # ax.axis("off")
        ax.grid()
        ax.set_axisbelow(True)
        
        ax.set_xlim((-10,15))
        ax.set_ylim((-15,10))
        ax.set_aspect('equal')
        ax.set_title("EFK Localization with Landmarks")
        plt.savefig("plots/prob1_plot.png", dpi=300)
        plt.show()
    


            

# Unit Tests:
sim = EKFSim()
sim.run()
sim.viz()
