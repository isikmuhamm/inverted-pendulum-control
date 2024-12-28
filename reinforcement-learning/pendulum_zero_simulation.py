import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pendulum_nonlinear_model import PendulumEnvironment

"""
Bu kodda pendulum_nonlinear_model.py dosyasındaki PendulumEnvironment sınıfı kullanılarak sisteme hiçbir kuvvet uygulanmayacak şekilde simülasyon gerçekleştirilmiştir.
Theta = 0 açısı aşağı kararlı denge durumunu temsil eder. Theta = π açısı yukarı kararlı denge durumunu temsil eder.

This code uses the PendulumEnvironment class from pendulum_nonlinear_model.py file to simulate the system without applying any force.
Theta = 0 angle represents the downward stable equilibrium position. Theta = π angle represents the upward stable equilibrium position.
"""


SIMULATION_DURATION = 400

if __name__ == "__main__":
    # Örnek kullanım:
    env = PendulumEnvironment()

    # Başlangıç durumu tanımlamaları
    duration = SIMULATION_DURATION
    time_series = int(duration/env.time_step)
    force = np.zeros(time_series)   
    states_history = []
    state = env.initial_state
    state = np.reshape(state, [1, 4])
    initial_state = state
    print(f"Başlangıç durumu: {state}")
    states_history.append(state)

     # Simülasyonu çalıştır
    for i in range(time_series - 1):
        next_state = (env.step(state[0], force[i]))
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        states_history.append(state)
    
    # Sonuçları kaydet
    save_folder = "reinforcement-learning"
    np.save(f"{save_folder}/states_zero.npy", states_history)
    
    print(f"Simülasyon kuvvet uygulanmaksızın tamamlandı. Sonuçlar kaydedildi. Zaman adımı {env.time_step} saniyedir. Başlangıç koşulu {initial_state} şeklindedir.")