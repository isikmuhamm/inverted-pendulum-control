import numpy as np
import random
from collections import deque
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parametreler
M = 0.5    # Cart mass (kg)
m = 0.2    # Pendulum mass (kg)
b = 0.1    # Coefficient of friction (N/m/sec)
l = 0.3    # Length to pendulum center of mass (m)
I = 0.006  # Moment of inertia of the pendulum (kg.m^2)
g = 9.81   # Gravitational acceleration (m/s^2)

# Diferansiyel denklem fonksiyonu
def pendulum_dynamics(state, t, M, m, b, l, I, g, F):
    x, x_dot, theta, theta_dot = state
    Sx = np.sin(theta)
    Cx = np.cos(theta)
    D = m * l * l * (M + m * (1 - Cx**2))

    x_ddot = (1 / D) * (-m**2 * l**2 * g * Cx * Sx + m * l**2 * (m * l * theta_dot**2 * Sx - b * x_dot)) + m * l * l * (1 / D) * F
    theta_ddot = (1 / D) * ((m + M) * m * g * l * Sx - m * l * Cx * (m * l * theta_dot**2 * Sx - b * x_dot)) - m * l * Cx * (1 / D) * F

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Pendulum step fonksiyonu (Bir adımda durumu hesapla)
def pendulum_step(state, F, time_step):
    t = [0, time_step]
    next_state = odeint(pendulum_dynamics, state[0], t, args=(M, m, b, l, I, g, F))
    return next_state[-1]  # Sonraki durum

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.batch_size = 32
        self.target_update_counter = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))  # İlk katman olarak Input kullanıyoruz
        model.add(Dense(64, activation='relu'))  # Artık input_dim kullanmamıza gerek yok
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Rastgele aksiyon
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Maksimum Q-değerini döndür

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:    
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Parametreler
state_size = 4  # [x, x_dot, theta, theta_dot]
action_size = 9  # [-10, -5, 0, +5, +10]
time_step = 0.1  # 20 ms zaman aralığı
force_values = np.linspace(-20, 20, action_size)  # Kuvvetler


# Ajanı oluştur
agent = DQNAgent(state_size, action_size)

# Simülasyon parametreleri
episodes = 1000  # Eğitim bölümleri
max_steps = 200  # Her bölümde maksimum adım
initial_state = [0.0, 0.0, np.pi/6, 0.0]  # Başlangıç durumu
states = []  # Durumları kaydetmek için bir liste

for e in range(episodes):
    state = np.array(initial_state)
    done = False
    total_reward = 0

    state = np.reshape(state, [1, 4])
    
    for step in range(max_steps):
        action = agent.act(state)
        force = force_values[action]
        next_state = pendulum_step(state, force, time_step)
        next_state = np.reshape(next_state, [1, 4])
        
        # Ödül fonksiyonu
        x, x_dot, theta, theta_dot = next_state[0]     
        theta_deg = np.degrees(abs(theta))  # Theta açısını dereceye çeviriyoruz.
        
        reward = np.cos(theta_deg)

        #if theta_deg < 10:
        #    reward = 10 - (theta ** 2) * 0.1 - (theta_dot ** 2) * 0.01  # Dik pozisyona yaklaştıkça pozitif ödül artıyor.
        #else:
        #    reward = -((theta**2) * 0.5 + abs(theta_dot) * 0.1)  # Dik pozisyondan uzaklaştıkça negatif ödül alıyor.

        # Eğer açı -10 ile 10 derece arasındaysa sayaç artar
        if theta_deg <= 15:
            consecutive_steps += 1
        else:
            consecutive_steps = 0  # Eğer açı bu sınırların dışına çıkarsa sayaç sıfırlanır

        # Sayaç 3'e ulaştığında done True olur
        done = (consecutive_steps >= 3) 

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Durumu kaydet
        states.append(state[0])

        if done or (step >= (max_steps-1)):
            print(f"episode: {e}/{episodes}, score: {total_reward}, epsilon: {agent.epsilon}, done: {done}")
            break

    agent.replay()
    
    if e % 10 == 0:
        agent.update_target_model() #Her 10 adımda bir modelin ağırlıkları güncelleniyor.
    if e % 100 == 0 and e>99:
        agent.learning_rate *= 0.5
        agent.epsilon *= 0.9
        print(f"%%% Parametre güncellemesi yapıldı. Learning rate: {agent.learning_rate}, Epsilon: {agent.epsilon} %%%")


# Sonuçları bir grafik üzerinde inceleyelim
def plot_results(states):
    states = np.array(states)  # Listeyi numpy dizisine çevir
    x = states[:, 0]      # Arabadaki pozisyon
    x_dot = states[:, 1]  # Arabadaki hız
    theta = states[:, 2]  # Açısal pozisyon
    theta_dot = states[:, 3]  # Açısal hız

    # Durum dizisini TXT dosyası olarak kaydet
    np.savetxt('pendulum_data.txt', states, delimiter=',', header='x,x_dot,theta,theta_dot', comments='')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x)
    plt.title("x (Araba Pozisyonu)")

    plt.subplot(2, 2, 2)
    plt.plot(x_dot)
    plt.title("x_dot (Araba Hızı)")

    plt.subplot(2, 2, 3)
    plt.plot(theta)
    plt.title("theta (Açı)")

    plt.subplot(2, 2, 4)
    plt.plot(theta_dot)
    plt.title("theta_dot (Açısal Hız)")

    plt.tight_layout()
    plt.show()

plot_results(states)
