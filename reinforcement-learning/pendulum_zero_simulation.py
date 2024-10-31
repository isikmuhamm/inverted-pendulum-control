import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pendulum_nonlinear_model import PendulumEnvironment

if __name__ == "__main__":
    # Örnek kullanım:
    env = PendulumEnvironment()

    # Başlangıç durumu tanımlamaları
    initial_state = [0.0, 0.0, np.random.uniform(-np.pi/18, np.pi/18), 0.0]
    duration = 400
    time_series = int(duration/env.time_step)
    force = np.zeros(time_series)   
    states = np.zeros((time_series, 4))
    states[0] = initial_state

    # Simülasyonu çalıştır
    for i in range(time_series - 1):
        states[i + 1] = env.step(states[i], force[i])
    
    # Sonuçları kaydet
    save_folder = "reinforcement-learning"
    #np.save(f"{save_folder}/states_zero.npy", states)
    
    print(f"Simülasyon kuvvet uygulanmaksızın tamamlandı. Sonuçlar kaydedildi. Zaman adımı {env.time_step} saniyedir.")