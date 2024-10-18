import numpy as np
import control as ctrl
import csv

# Sistem parametreleri
M, m, b, I, g, l = 0.5, 0.2, 0.1, 0.006, 9.8, 0.3
p = I*(M+m)+M*m*l**2

# Durum uzayı matrisleri
A = np.array([
    [0, 1, 0, 0],
    [0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
    [0, 0, 0, 1],
    [0, -(m*l*b)/p, m*g*l*(M+m)/p, 0]
])

B = np.array([[0], [(I+m*l**2)/p], [0], [m*l/p]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
D = np.array([[0], [0]])

# LQR kontrolcü tasarımı
Q = np.dot(C.T, C)
R = 1
K, _, _ = ctrl.lqr(A, B, Q, R)

# Kapalı çevrim sistem
Ac = A - np.dot(B, K)
sys_cl = ctrl.ss(Ac, B, C, D)

# Simülasyon
t = np.arange(0, 5, 0.01)
x0 = np.array([[0], [0], [0], [0]])
results = []

for k in range(50):
    r = 0.1 * (k+1) * np.ones((1, t.size))
    response = ctrl.forced_response(sys_cl, T=t, U=r, X0=x0)
    y = response.outputs
    x = response.states
    
    for i in range(len(t)):
        x_i = x[:, i].reshape(-1, 1)
        r_i = r[:, i].reshape(-1, 1)
        xdot = np.dot(A, x_i) + np.dot(B, r_i)
        F = (M+m)*xdot[1, 0] + l*m*xdot[3, 0]
        results.append([t[i], x[0, i], x[1, i], x[2, i], x[3, i], r[0, i], F])

# Sonuçları CSV dosyasına kaydet
with open('supervised-learning/pendulum_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['t', 'x', 'x_dot', 'theta', 'theta_dot', 'r', 'F'])
    writer.writerows(results)

print("Simülasyon tamamlandı. Veriler 'pendulum_data.csv' dosyasına kaydedildi.")