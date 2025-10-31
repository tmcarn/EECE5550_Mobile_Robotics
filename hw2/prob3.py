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
        This function uses the motion model on a set of particles, all with slightly different wheel velocities
        
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
        This function takes in one particle, with one set of wheel velocities, 
        and returns the new postion based on dead reckoning

        true_wheel_vels_i: [true_phi_r, true_phi_l]
        '''
        x_dot = self.r/2 * (np.sum(true_wheel_vels_i))
        theta_dot = self.r/self.w * (true_wheel_vels_i[0]-true_wheel_vels_i[1])

        omega_dot = np.zeros((3,3))
        omega_dot[0,1] = -theta_dot
        omega_dot[0,2] = x_dot
        omega_dot[1,0] = theta_dot

        dead_rek = x_i @ expm(dt * omega_dot)

        return dead_rek
    
    def measurement_update(self, X_prior, obs_measurement):
        '''
        Resamples particles based on the probability of obtaining the observed measurement from that positon
        '''
        prior_position = X_prior[:, :2, 2] # Extracts translation from each SE(2) matrix
        innovation = np.linalg.norm(obs_measurement - prior_position, axis=1) # Calculates l2 norm between observed measurement and predicted measurement
        importance_factor = np.exp(-(innovation ** 2) / (2 * (self.sigma_p ** 2)))
        # Normalize Importance Factors
        importance_factor = importance_factor/np.sum(importance_factor)

        # Multinomial Resampling
        cumsum = np.cumsum(importance_factor)
        indices = np.searchsorted(cumsum, np.random.uniform(0, 1, self.num_samples)) 

        # Posterior updated based on sampling of prior
        X_posterior = X_prior[indices]

        return X_posterior

    def run_propagation(self, cmd_wheel_vels, time_steps):
        '''
        Runs dead reckoning with out any filtering
        '''
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
    
    def run_filter(self, cmd_wheel_vels, time_steps):
        prev_t = 0
        X_posterior_history = np.zeros((time_steps.shape[0], self.num_samples, 3, 3))

        for i,t in enumerate(time_steps):
            self.X_prior = self.propagation(self.X_posterior, prev_t, t, cmd_wheel_vels)
            self.X_posterior = self.measurement_update(self.X_prior, self.measurements[i])
            X_posterior_history[i] = self.X_posterior
            prev_t = t
        
        return X_posterior_history


    def calculate_stats(self, X):
        '''
        Given particle set X, returns the mean and covariance of the set
        '''
        X_pos = X[:, 0:2, 2]
        mean = np.mean(X_pos, axis=0)

        centered = X_pos - mean
        cov = (centered.T @ centered) / self.num_samples

        return mean, cov
    
    def save_stats_to_file(self, mean, cov, time_step, path):
        '''
        Saves mean and covariance statistics to a text file
        '''     
        with open(path, 'a') as f:
            f.write("\n")
            f.write(f"Statistics at t={time_step}\n")
            f.write(f"Number of particles: {self.num_samples}\n\n")
            
            f.write(f"Mean Position:\n")
            f.write(f"  x: {mean[0]:.6f}\n")
            f.write(f"  y: {mean[1]:.6f}\n\n")
            
            f.write(f"Covariance Matrix:\n")
            f.write(f"  [{cov[0,0]:.6f}, {cov[0,1]:.6f}]\n")
            f.write(f"  [{cov[1,0]:.6f}, {cov[1,1]:.6f}]\n\n")
            
            f.write(f"Variance:\n")
            f.write(f"  σ²_x: {cov[0,0]:.6f}\n")
            f.write(f"  σ²_y: {cov[1,1]:.6f}\n")
            f.write(f"  σ_xy: {cov[0,1]:.6f}\n")

            f.write(f"{"_"*50}")
            f.write("\n")

    def plot_particles(self, X_history, time_steps, title, fname):
        
        # Initalize Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(0,0, label="t=0")

        stats_fname = fname.split('_')[0]+"_stats.txt"
        stats_path = os.path.join("stats", stats_fname)
        open(stats_path, 'w').close() # Erase Previous Content

        for i in range(X_history.shape[0]):
            X = X_history[i]

            mean, cov = self.calculate_stats(X)
            self.save_stats_to_file(mean, cov, time_steps[i], stats_path)

            x = X[:, 0, 2]
            y = X[:, 1, 2]
            theta = np.arctan2(X[:, 1, 0], X[:, 0, 0])
    
            # Calculate heading direction
            dx = np.cos(theta)
            dy = np.sin(theta)

            ax.scatter(x, y, label=f"t={time_steps[i]}", alpha=0.1)
            ax.quiver(x, y, dx, dy, alpha=0.05, scale_units='xy', angles='xy', width=0.003) # Plots Particle Orientation
            ax.scatter(mean[0], mean[1], alpha=1, c='cyan')
        
        ax.axis('equal')
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join("plots", fname), dpi=300)
        plt.show()


# Plot particle poses at t=10 after applying motion model
pf = ParticleFilter()
cmd_vels = np.array([2.0, 1.5]) # [right_wheel, left_wheel]
time_steps = np.array([10])
X_prior_history = pf.run_propagation(cmd_vels, time_steps)
title = "Particle Poses at t=10 after applying Motion Model"
fname = "prob3e_particles_motion_model.png"
pf.plot_particles(X_prior_history, time_steps, title, fname)

# Plot particle poses at for t=[5,10,15,20] after applying motion model recursively
pf = ParticleFilter()
cmd_vels = np.array([2.0, 1.5]) # [right_wheel, left_wheel]
time_steps = np.array([5,10,15,20])
X_prior_history = pf.run_propagation(cmd_vels, time_steps)
title = "Particle Poses at t=[5,10,15,20] after applying Motion Model Recursively"
fname = "prob3f_particles_motion_model.png"
pf.plot_particles(X_prior_history, time_steps, title, fname)

# Plot particle poses at for t=[5,10,15,20] after applying motion model recursively
pf = ParticleFilter()
cmd_vels = np.array([2.0, 1.5]) # [right_wheel, left_wheel]
time_steps = np.array([5,10,15,20])
X_posterior_history = pf.run_filter(cmd_vels, time_steps)
title = "Particle Poses at t=[5,10,15,20] after applying Particle Filter"
fname = "prob3g_particle_filter.png"
pf.plot_particles(X_posterior_history, time_steps, title, fname)
pf.calculate_stats(X_posterior_history[0])