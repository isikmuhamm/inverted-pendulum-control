import numpy as np
from scipy.integrate import odeint

"""
Bu kodda sistem modeli olarak https://ctms.engin.umich.edu/CTMS/?example=InvertedPendulum&section=SystemModeling adresindeki model kullanılmıştır.
Theta = 0 açısı aşağı kararlı denge durumunu temsil eder. Theta = π açısı yukarı kararlı denge durumunu temsil eder.

This code uses the system model from https://ctms.engin.umich.edu/CTMS/?example=InvertedPendulum&section=SystemModeling as for the system model.
Theta = 0 angle represents the downward stable equilibrium position. Theta = π angle represents the upward stable equilibrium position.
"""

class PendulumEnvironment:
    def __init__(self):
        # Sistem parametreleri
        self.M = 0.5    # Cart mass (kg)
        self.m = 0.2    # Pendulum mass (kg)
        self.b = 0.1    # Coefficient of friction (N/m/sec)
        self.l = 0.3    # Length to pendulum center of mass (m)
        self.I = 0.006  # Moment of inertia (kg.m^2)
        self.g = 9.81   # Gravitational acceleration (m/s^2)
        self.time_step = 0.035  # 35ms zaman aralığı
        self.action_size = 9
        self.state_size = 4
        self.force_values = np.linspace(-1, 1, self.action_size)*10
        self.attack_values = np.linspace(-1, 1, self.action_size)*3
        
    @property
    def initial_state(self):
        return np.array([
            np.random.uniform(-0.1, 0.1),          # x
            np.random.uniform(-0.1, 0.1),          # x_dot
            np.random.uniform(np.pi-np.pi/18, np.pi+np.pi/18),  # theta
            np.random.uniform(-0.1, 0.1)           # theta_dot
        ])   
    def dynamics(self, state, t, F):
        x, x_dot, theta, theta_dot = state
        Sx = np.sin(theta)
        Cx = np.cos(theta)
        
        # Atalet momentini içeren D matrisi
        D = (self.M + self.m) * (self.I + self.m * self.l**2) - (self.m * self.l * Cx)**2

        # İyileştirilmiş hareket denklemleri
        x_ddot = (1 / D) * (
            (self.I + self.m * self.l**2) * (F - self.b * x_dot + self.m * self.l * theta_dot**2 * Sx) +
            (self.m**2 * self.l**2 * self.g * Sx * Cx)
        )
        
        # Açısal hareket denklemi (theta için)
        theta_ddot = (1 / D) * (
            -self.m * self.l * Cx * (F - self.b * x_dot + self.m * self.l * theta_dot**2 * Sx) +
            (self.M + self.m) * (-self.m * self.g * self.l * Sx)
        )

        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
   
    def step(self, state, F):
        t = [0, self.time_step]
        next_state = odeint(self.dynamics, state, t, args=(F,))[-1]
        
        # Theta'yı [-π, π] aralığında normalize et
        theta = ((next_state[2] + np.pi) % (2 * np.pi)) - np.pi
        
        return [next_state[0], next_state[1], theta, next_state[3]]