import numpy as np
from matplotlib import pyplot as plt

class AltitudeControl:
    def __init__(self, gains=(0,0,0), conv_time=None, zeta=None, actuator_noise=False):
        self.mass = 0.065 #kg
        self.thrust_coeff = 5.276e-4
        self.g = 9.81 # m/s^2

        self.dt = 0.01 # seconds (100 Hz)
        self.n_steps = 1_500 # 15 seconds of run

        self.h_pos = 0
        self.h_vel = 0
        self.h_acc = 0

        self.setpoint = 1 # meters

        self.prev_error = None
        self.int_error = 0

        self.Kp, self.Ki, self.Kd = gains

        if conv_time is not None:
            self.calculate_gains(conv_time, zeta)

        self.actuator_noise = actuator_noise

    def calculate_gains(self, conv_time, zeta):
        # 0 < zeta < 1 : UNDERDAMPED
        # zeta = 1 : Critically Damped
        # zeta > 1 : OVERDAMPED

        # Calculate Kp and Kd
        self.zeta = zeta
        self.omega = 3 / (conv_time * zeta)

        self.Kp = (self.omega ** 2 * self.mass) / (4 * self.thrust_coeff)
        self.Kd = (self.zeta * self.omega * self.mass) / (2 * self.thrust_coeff)


    def state_transition_model(self, u):
        if self.actuator_noise:
            u *= 0.95
        # Calculate new acceleration based on control input u
        self.h_acc = (4 * self.thrust_coeff * u) / self.mass - self.g

        # Integrate acceleration to update velocity & position
        self.h_vel += self.h_acc * self.dt
        self.h_pos += self.h_vel * self.dt

        return self.h_acc, self.h_vel, self.h_pos
    
    def control_step(self):
        e = self.setpoint - self.h_pos

        # Integral Error
        self.int_error += e * self.dt

        # Derrivative Error
        derv_error = 0
        if self.prev_error is not None:
            derv_error = (e - self.prev_error) / self.dt

        grav_offset = (self.mass * self.g) / (4 * self.thrust_coeff)

        u = (self.Kp * e) + (self.Ki * self.int_error) + (self.Kd * derv_error) + grav_offset

        # Update previous error for next control step
        self.prev_error = e

        return u
    
    def run_controller(self):
        h_pos_hist = [] 
        h_acc_hist = []
        for i in range(self.n_steps):
            # Generate Actuation
            u = self.control_step()
            # Update Position based on actuation
            self.state_transition_model(u)

            # Update history for plotting
            h_pos_hist.append(self.h_pos)
            h_acc_hist.append(self.h_acc)

        x_axis = np.arange(self.n_steps) * self.dt

        return np.array(h_pos_hist), np.array(h_acc_hist), x_axis
    
def format_plot(ax1, ax2, title1, title2, path):
    # Plot Setpoint
    ax1.axhline(y=1, color='r', linestyle='--', label='Setpoint')

    # Format Plot
    ax1.set_title(title1)
    ax1.set_ylabel("Altitude ($m$)")
    ax1.set_xlabel("seconds")

    ax2.set_title(title2)
    ax2.set_ylabel("Acceleration ($m/s^2$)")
    ax2.set_xlabel("seconds")

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    ax1.legend()
    ax2.legend()

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    

# ______________________________________________________________________________

# Part A
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,9)) # Create a figure with two subplots side by side

Kps = [5, 15, 50]
for Kp in Kps:
    alt_ctrl = AltitudeControl((Kp, 0, 0)) # No integral or derivative terms
    h_pos_hist, h_acc_hist, x_axis = alt_ctrl.run_controller()

    # Plot Position over time
    ax1.plot(x_axis, h_pos_hist, label=f"Kp={Kp}")
    ax2.plot(x_axis, h_acc_hist, label=f"Kp={Kp}")

t1 = "P Controller: Altitude over Time"
t2 = "P Controller: Acceleration over Time"
format_plot(ax1, ax2, t1, t2, "hw4/plots/PControllerPlot.png")
# ______________________________________________________________________________

# Part B: Design a PD Controller that converges in ~3 seconds and is underdamped

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,9)) # Create a figure with two subplots side by side

alt_ctrl = AltitudeControl(conv_time=3, zeta=0.5) # UNDERDAMPED
h_pos_hist, h_acc_hist, x_axis = alt_ctrl.run_controller()

# Plot Position over time
ax1.plot(x_axis, h_pos_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")
ax2.plot(x_axis, h_acc_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")

t1 = rf"Underdamped PD Controller: Altitude over Time ($\zeta = {alt_ctrl.zeta}$, $\omega = {alt_ctrl.omega}$)"
t2 = rf"Underdamped PD Controller: Acceleration over Time ($\zeta = {alt_ctrl.zeta}$, $\omega = {alt_ctrl.omega}$)"
format_plot(ax1, ax2, t1, t2, "hw4/plots/PD_Underdamped.png")
# ______________________________________________________________________________

# Part C: Design a PD Controller that converges in ~3 seconds and is overderdamped

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,9)) # Create a figure with two subplots side by side

alt_ctrl = AltitudeControl(conv_time=3, zeta=1.5) # OVERDAMPED
h_pos_hist, h_acc_hist, x_axis = alt_ctrl.run_controller()

# Plot Position over time
ax1.plot(x_axis, h_pos_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")
ax2.plot(x_axis, h_acc_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")

t1 = rf"Overdamped PD Controller: Altitude over Time ($\zeta = {alt_ctrl.zeta:.2f}$, $\omega = {alt_ctrl.omega:.2f}$)"
t2 = rf"Overdamped PD Controller: Acceleration over Time ($\zeta = {alt_ctrl.zeta:.2f}$, $\omega = {alt_ctrl.omega:.2f}$)"
format_plot(ax1, ax2, t1, t2, "hw4/plots/PD_Overdamped.png")
# ______________________________________________________________________________

# Part D(1): PD Controller with Noisy Actuation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,9)) # Create a figure with two subplots side by side

alt_ctrl = AltitudeControl(conv_time=3, zeta=0.5, actuator_noise=True) # UNDERDAMPED
h_pos_hist, h_acc_hist, x_axis = alt_ctrl.run_controller()

# Plot Position over time
ax1.plot(x_axis, h_pos_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")
ax2.plot(x_axis, h_acc_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Kd={alt_ctrl.Kd:.2f}")

t1 = "PD Controller: Altitude over Time (Noisy Actuation)"
t2 = "PD Controller: Acceleration over Time (Noisy Actuation)"
format_plot(ax1, ax2, t1, t2, "hw4/plots/PD_Noisy.png")
# ______________________________________________________________________________

# Part D(2): PID Controller with Noisy Actuation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,9)) # Create a figure with two subplots side by side

# Set a non-negative integral gain
alt_ctrl = AltitudeControl((0,30,0), conv_time=3, zeta=0.5, actuator_noise=True)
h_pos_hist, h_acc_hist, x_axis = alt_ctrl.run_controller()

# Plot Position over time
ax1.plot(x_axis, h_pos_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Ki={alt_ctrl.Ki:.2f}, Kd={alt_ctrl.Kd:.2f}")
ax2.plot(x_axis, h_acc_hist, label=f"Kp={alt_ctrl.Kp:.2f}, Ki={alt_ctrl.Ki:.2f}, Kd={alt_ctrl.Kd:.2f}")

t1 = "PID Controller: Altitude over Time (Noisy Actuation)"
t2 = "PID Controller: Acceleration over Time (Noisy Actuation)"
format_plot(ax1, ax2, t1, t2, "hw4/plots/PID_Noisy.png")


