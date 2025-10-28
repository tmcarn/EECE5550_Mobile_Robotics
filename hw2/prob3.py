import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt
import os

class ParticleFilter():
    def __init__(self, r=0.25, w=0.5, sigma_l=0.05, sigma_r=0.05, sigma_p=0.1):

        # Hardware Constants
        self.r = r
        self.w = w

        self.motion_cov = np.zeros((2,2))
        self.motion_cov[0,0] = sigma_r ** 2
        self.motion_cov[1,1] = sigma_l ** 2

        self.sigma_p = sigma_p
        self.measurement_cov = (sigma_p ** 2) * np.eye(2)

        self.time_steps = np.array([5, 10, 15, 20])

        # Initial Sampling
        self.num_samples = 1000

        self.X_posterior = np.tile(np.eye(3), (self.num_samples, 1, 1)) # All particles initialized at x,y,theta = 0
        self.X_prior = np.zeros((self.num_samples, 3, 3))

        self.X_posterior_history = np.zeros((self.time_steps.shape[0], self.num_samples, 3, 3))
        self.X_prior_history = np.zeros((self.time_steps.shape[0], self.num_samples, 3, 3))

        # Imperical Measurements
        measurements = [[1.6561, 1.2847], [1.0505, 3.1059], [-0.9075, 3.2118], [-1.6450, 1.1978]]
        self.measurements = np.array(measurements)


    def propagation(self, X_posterior, t1, t2, cmd_wheel_vels):
        '''
        Inputs:
             X_posterior : Esitmate poses from t-1

        Outputs:
            X_prior : Deadreckoning for t 
        '''
        vel_noise = np.random.multivariate_normal(np.zeros(2), self.motion_cov, self.num_samples)
        true_wheel_vels = cmd_wheel_vels + vel_noise
        dt = t2 - t1

        X_prior = np.zeros((self.num_samples, 3, 3))

        for i in range(X_posterior.shape[0]):
            pose = self.motion_model(true_wheel_vels[i], X_posterior[i], dt)
            X_prior[i] = pose

        return X_prior

    def motion_model(self, true_wheel_vels_i, x_i, dt):
        '''
        true_wheel_vels_i: [true_phi_r, true_phi_l]
        '''
        x_dot = self.r/2 * (np.sum(true_wheel_vels_i))
        theta_dot = self.r/self.w * (true_wheel_vels_i[0]-true_wheel_vels_i[1])

        omega_dot = np.zeros((3,3))
        omega_dot[0,1] = -theta_dot
        omega_dot[0,2] = x_dot
        omega_dot[1,0] = theta_dot

        dead_rek = x_i + expm(dt * omega_dot)

        return dead_rek
    
    def measurement_update(self, X_prior, obs_measurement):
        # measurement_noise = np.random.multivariate_normal(np.zeros(2), self.measurement_cov)
        # obs_measurement = true_measurement + measurement_noise

        prior_position = X_prior[:, :2, 2] # Extracts translation from each SE(2) matrix
        innovation = np.linalg.norm(obs_measurement - prior_position, axis=1) # Calculates l2 norm between observed measurement and predicted measurement
        importance_factor = -(np.exp(innovation) ** 2) / (2 * (self.sigma_p ** 2))

        # Normalize Importance Factors
        importance_factor = importance_factor/np.sum(importance_factor)

        # Multinormal Resampling
        cumsum = np.cumsum(importance_factor)
        indices = np.searchsorted(cumsum, np.random.uniform(0, 1, self.num_samples)) 

        # Posterior updated based on sampling of prior
        X_posterior = X_prior[indices]

        return X_posterior

    def run_propagation(self, cmd_wheel_vels, time_steps):
        prev_t = 0

        # Initial Position
        prev_X_prior = np.tile(np.eye(3), (self.num_samples, 1, 1)) # All particles initialized at x,y,theta = 0
        X_prior_history = np.zeros((time_steps.shape[0], self.num_samples, 3, 3))

        for i,t in enumerate(time_steps):
            X_prior = self.propagation(prev_X_prior, prev_t, t, cmd_wheel_vels)
            X_prior_history[i] = X_prior
            prev_X_prior = X_prior
            prev_t = t

        return X_prior_history
            

    def calculate_stats(self, X):
        pass

    def plot_particles(self, X_history, time_steps, title, fname):
        # Initalize Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(0,0, s=200, label="t=0")

        for i in range(X_history.shape[0]):
            X = X_history[i]

            x = X[:, 0, 2]
            y = X[:, 1, 2]
            theta = np.arctan2(X[:, 1, 0], X[:, 0, 0])
    
            # Calculate heading direction
            dx = np.cos(theta)
            dy = np.sin(theta)

            ax.scatter(x, y, label=f"t={time_steps[i]}", alpha=0.1)
            # ax.quiver(x, y, dx, dy,
            #     alpha=0.25, scale_units='xy', 
            #     angles='xy', width=0.003)
        
        ax.axis('equal')
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join("plots", fname))
        plt.show()




# Plot particle poses at t=10 after applying motion model
pf = ParticleFilter()
cmd_vels = np.array([2.0, 1.5]) # [right_wheel, left_wheel]
time_steps = np.array([10])
X_prior_history = pf.run_propagation(cmd_vels, time_steps)
title = "Particle Poses at t=10 after applying Motion Model"
fname = "Q3E_particles_motion_model.png"
pf.plot_particles(X_prior_history, time_steps, title, fname)


# Plot particle poses at for t=[5,10,15,20] after applying motion model recursively
pf = ParticleFilter()
cmd_vels = np.array([2.0, 1.5]) # [right_wheel, left_wheel]
time_steps = np.array([5,10,15,20])
X_prior_history = pf.run_propagation(cmd_vels, time_steps)
title = "Particle Poses at t=[5,10,15,20] after applying Motion Model Recursively"
fname = "Q3F_particles_motion_model.png"
pf.plot_particles(X_prior_history, time_steps, title, fname)
