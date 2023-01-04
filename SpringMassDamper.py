
# Standard library imports
import os
# Conda imports
from matplotlib import pyplot as plt
import numpy as np


class SpringMassDamper():
    def __init__(self, position, velocity, acceleration, dt = 0.002, mass = 1, k = 2.5, c = 0.3):
        self.position = position # initial position in meters
        self.velocity = velocity # initial velocity in m/s
        self.acceleration = acceleration # initial acceleration in m/s^2
        self.dt = dt  # time step in seconds
        self.mass = mass  # mass of object in kg
        self.k = k  # spring coefficient in N/m
        self.c = c  # damper coefficient in N/m/s

    def time_step(self):
        spring_force = self.k * self.position # Fs = k * x
        damper_force = self.c * self.velocity # Fb = c * x'

        self.acceleration = - (spring_force + damper_force) / self.mass
        self.velocity += (self.acceleration * self.dt) # Integral(a) = v
        self.position += (self.velocity * self.dt) # Integral(v) = x

    def collect_data(self, steps):
        time = 0
        self.pos_hist = []
        self.vel_hist = []
        self.acc_hist = []
        self.time_hist = []
        
        for step in range(steps):
            self.time_step()
            self.pos_hist.append(self.position)
            self.vel_hist.append(self.velocity)
            self.acc_hist.append(self.acceleration)
            self.time_hist.append(time)
            time += self.dt

        return (self.pos_hist, self.vel_hist, self.acc_hist, self.time_hist)

    def plot_all_data(self, pos_hist, vel_hist, acc_hist, time_hist, save_plot=True):
        fig =  plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        # position plot
        axes = fig.add_subplot(1, 3, 1)
        axes.plot(time_hist, pos_hist)
        axes.set_title('Lateral Position Over Time')
        axes.set_xlabel('time (s)')
        axes.set_ylabel('position (m)')

        # velocity plot
        axes = fig.add_subplot(1, 3, 2)
        axes.plot(time_hist, vel_hist)
        axes.set_title('Velocity Over Time')
        axes.set_xlabel('time (s)')
        axes.set_ylabel('velocity (m/s)')

        # acceleration over time
        axes = fig.add_subplot(1, 3, 3)
        axes.plot(time_hist, acc_hist)
        axes.set_title('Acceleration Over Time')
        axes.set_xlabel('time (s)')
        axes.set_ylabel('acceleration (m/s^2)')

        plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
        
        if save_plot:
            plt.savefig(os.path.join('Images', 'position_velocity_acceleration_history.png'))
        else:
            plt.show()

if __name__=="__main__":
    # Constants
    dt = 0.01 # sampling period in seconds
    MASS = 1 # mass
    SPRING_COEFFICIENT = 2.5 # spring coefficient (k)
    DAMPING_COEFFICIENT = 0.3 # damping coefficient (b)

    # Initial values
    INITIAL_POSITION = 15
    INITIAL_VELOCITY = 0
    INITIAL_ACCELERATION = 0

    TIME_STEPS = 2000

    cart = SpringMassDamper(position = INITIAL_POSITION, 
                                velocity = INITIAL_VELOCITY, 
                                acceleration = INITIAL_ACCELERATION, 
                                dt = dt, 
                                mass = MASS, 
                                k = SPRING_COEFFICIENT,
                                c = DAMPING_COEFFICIENT)
    
    pos_hist, vel_hist, acc_hist, time_hist = cart.collect_data(TIME_STEPS)
    cart.plot_all_data(pos_hist, vel_hist, acc_hist, time_hist)
    
    pass