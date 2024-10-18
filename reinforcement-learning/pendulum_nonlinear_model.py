import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametreler
M = 0.5    # Cart mass (kg)
m = 0.2    # Pendulum mass (kg)
b = 0.1    # Coefficient of friction (N/m/sec)
l = 0.3    # Length to pendulum center of mass (m)
I = 0.006  # Moment of inertia of the pendulum (kg.m^2)
g = 9.81   # Gravitational acceleration (m/s^2)

# Pendulum dinamiği (diferansiyel denklemler)
def pendulum_dynamics(y, t, M, m, b, l, I, g, F):
    x, x_dot, theta, theta_dot = y
    Sx = np.sin(theta)
    Cx = np.cos(theta)
    D = m * l * l * (M + m * (1 - Cx**2))

    x_ddot = (1 / D) * (-m**2 * l**2 * g * Cx * Sx + m * l**2 * (m * l * theta_dot**2 * Sx - b * x_dot)) + m * l * l * (1 / D) * F
    theta_ddot = (1 / D) * ((m + M) * m * g * l * Sx - m * l * Cx * (m * l * theta_dot**2 * Sx - b * x_dot)) - m * l * Cx * (1 / D) * F

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Tek adımda durumu hesaplayan fonksiyon (odeint kullanarak)
def pendulum_step_odeint(state, time_step, force):
    t = [0, time_step]  # Zaman adımı
    next_state = odeint(pendulum_dynamics, state, t, args=(M, m, b, l, I, g, force))
    return next_state[-1]  # Sadece bir sonraki adımı döndür

# Örnek kullanım:

# Zaman dizisi (0'dan 100'e kadar 1000 adım)
time = np.linspace(0, 1000, 10000)

# Kuvvet dizisi (0'lar dizisi)
force = np.zeros(len(time))

# Durumları tutacağımız dizi (4 sütunlu, time uzunluğunda)
states = np.zeros((len(time), 4))

# Başlangıç durumu: [x, x_dot, theta, theta_dot]
initial_state = [0.0, 0.0, 0.0, 0.5]
states[0] = initial_state

# Zaman adımı (time_step)
time_step = time[1] - time[0]

# Simülasyonu for döngüsü ile çalıştır
for i in range(len(time) - 1):
    states[i + 1] = pendulum_step_odeint(states[i], time_step, force[i])

# Durum dizisini TXT dosyası olarak kaydet
np.savetxt('pendulum_data.txt', states, delimiter=',', header='x,x_dot,theta,theta_dot', comments='')


# Grafiği oluşturma
plt.figure(figsize=(10, 8))

# Arabanın pozisyonu x
plt.subplot(4, 1, 1)
plt.plot(time, states[:, 0], color='b')
plt.title('Cart Position (x)')
plt.ylabel('Position [m]')

# Arabanın hızı x_dot
plt.subplot(4, 1, 2)
plt.plot(time, states[:, 1], color='g')
plt.title('Cart Velocity (x_dot)')
plt.ylabel('Velocity [m/s]')

# Sarkacın açısı theta
plt.subplot(4, 1, 3)
plt.plot(time, states[:, 2], color='r')
plt.title('Pendulum Angle (theta)')
plt.ylabel('Angle [rad]')

# Sarkacın açısal hızı theta_dot
plt.subplot(4, 1, 4)
plt.plot(time, states[:, 3], color='m')
plt.title('Pendulum Angular Velocity (theta_dot)')
plt.ylabel('Angular Velocity [rad/s]')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

x = states[:, 0]      # Arabanın pozisyonu
theta = states[:, 2]  # Sarkacın açısı (radyan cinsinden)

# Animasyon oluşturma
fig, ax = plt.subplots(figsize=(8, 6))

# Aracın ve sarkacın çizimi için sınırlar
ax.set_xlim([-2, 2])
ax.set_ylim([-1, 1.5])

# Aracın ve sarkacın görselleştirilmesi
cart, = ax.plot([], [], 'ks-', lw=10)  # Aracı temsil eden siyah bir kare
pendulum_line, = ax.plot([], [], 'ro-', lw=2)  # Sarkaç çizgisi

# Animasyonun başlangıç durumu
def init():
    cart.set_data([], [])
    pendulum_line.set_data([], [])
    return cart, pendulum_line

# Animasyon fonksiyonu (her kareyi çizen fonksiyon)
def animate(i):
    # Aracın pozisyonu
    cart_x = [x[i] - 0.1, x[i] + 0.1]  # Aracı bir kare olarak modelle
    cart_y = [0, 0]

    # Sarkaç ucu pozisyonu
    pendulum_x = [x[i], x[i] + l * np.sin(theta[i])]  # Sarkaç yatay pozisyonu
    pendulum_y = [0, l * np.cos(theta[i])]           # Sarkaç dikey pozisyonu

    # Yeni verileri çiz
    cart.set_data(cart_x, cart_y)
    pendulum_line.set_data(pendulum_x, pendulum_y)
    
    return cart, pendulum_line

# Animasyonu başlat
ani = FuncAnimation(fig, animate, frames=len(time), init_func=init, interval=20, blit=True)

plt.show()