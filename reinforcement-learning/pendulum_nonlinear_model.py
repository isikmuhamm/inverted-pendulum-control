import numpy as np
from scipy.integrate import odeint

class PendulumEnvironment:
    def __init__(self):
        # Sistem parametreleri
        self.M = 0.5    # Cart mass (kg)
        self.m = 0.2    # Pendulum mass (kg)
        self.b = 0.1    # Coefficient of friction (N/m/sec)
        self.l = 0.3    # Length to pendulum center of mass (m)
        self.I = 0.006  # Moment of inertia (kg.m^2)
        self.g = 9.81   # Gravitational acceleration (m/s^2)
        self.time_step = 0.1  # 200ms zaman aralığı
        
    def dynamics(self, state, t, F):
        x, x_dot, theta, theta_dot = state
        Sx = np.sin(theta)
        Cx = np.cos(theta)
        D = self.m * self.l * self.l * (self.M + self.m * (1 - Cx**2))

        x_ddot = (1 / D) * (-self.m**2 * self.l**2 * self.g * Cx * Sx + 
                           self.m * self.l**2 * (self.m * self.l * theta_dot**2 * Sx - self.b * x_dot)) + \
                 self.m * self.l * self.l * (1 / D) * F
        
        theta_ddot = (1 / D) * ((self.m + self.M) * self.m * self.g * self.l * Sx - 
                               self.m * self.l * Cx * (self.m * self.l * theta_dot**2 * Sx - self.b * x_dot)) - \
                     self.m * self.l * Cx * (1 / D) * F

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    """     def step(self, state, F):
        t = [0, self.time_step]
        return odeint(self.dynamics, state, t, args=(F,))[-1] """
    
    def step(self, state, F):
        t = [0, self.time_step]
        next_state = odeint(self.dynamics, state, t, args=(F,))[-1]
        
        # Theta'yı 0 ile 2π arasında tut
        theta = next_state[2] % (2 * np.pi)
        if theta > np.pi: theta -= 2 * np.pi
        
        return [next_state[0], next_state[1], theta, next_state[3]]