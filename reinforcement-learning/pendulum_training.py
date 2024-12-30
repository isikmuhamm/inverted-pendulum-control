import numpy as np
import random
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from pendulum_nonlinear_model import PendulumEnvironment

CONTINUE_TRAINING = False
POISSON_IMPATCS = False
save_folder = "reinforcement-learning"
max_len = 50000
poisson_lambda = 10

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Model parametreleri
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=max_len)
        self.gamma = 0.99    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.0001
        self.batch_size = 64
        
        # Modeller
        if CONTINUE_TRAINING:
            self.load_agent(f"{save_folder}")
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Double DQN implementasyonu
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        target_q_values = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * target_q_values[np.arange(len(next_actions)), next_actions] * (1 - dones)
        
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(self.batch_size), actions] = targets
        
        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Adaptive epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_agent(self, folder_path):
        """Ajanın tüm durumunu kaydet"""
        try:
            # Klasör yoksa oluştur
            os.makedirs(folder_path, exist_ok=True)
            
            # Modeli kaydet
            model_path = f'{folder_path}/pendulum_model.keras'
            self.model.save(model_path)
            print(f"Model başarıyla {model_path} konumuna kaydedildi.")
            
            # Ajanın durumu
            agent_state = {
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'memory': list(self.memory)
            }
            agent_state_path = f'{folder_path}/agent_state.npy'
            np.save(agent_state_path, agent_state)
            print(f"Ajan durumu başarıyla {agent_state_path} konumuna kaydedildi.")
        
        except Exception as e:
            print(f"Hata: Ajan kaydedilirken bir sorun oluştu. Hata mesajı: {e}")

    def load_agent(self, folder_path):
        """Ajanın tüm durumunu yükle"""
        try:
            # Model yolları
            model_path = f'{folder_path}/pendulum_model.keras'
            agent_state_path = f'{folder_path}/agent_state.npy'
            
            # Model dosyası mevcut mu?
            if not os.path.exists(model_path) or not os.path.exists(agent_state_path):
                raise FileNotFoundError(f"Model veya ajan durumu dosyası bulunamadı: {model_path} veya {agent_state_path}")
            
            # Modeli yükle
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
            print(f"Model başarıyla {model_path} konumundan yüklendi.")
            
            # Ajanın durumu
            agent_state = np.load(agent_state_path, allow_pickle=True).item()
            self.epsilon = agent_state.get('epsilon', self.epsilon)
            self.learning_rate = agent_state.get('learning_rate', self.learning_rate)
            self.memory = deque(agent_state.get('memory', []), maxlen=max_len)
            print(f"Ajan durumu başarıyla {agent_state_path} konumundan yüklendi.")
        
        except FileNotFoundError as fnf_error:
            print(f"Hata: {fnf_error}")
        except ValueError as ve:
            print(f"Hata: Ajan durumu dosyası bozuk veya eksik olabilir. Hata mesajı: {ve}")
        except Exception as exc:
            print(f"Bilinmeyen bir hata oluştu: {exc}")


def train(episodes=2000, max_steps=200):
    # Ajanın ve ortamın oluşturulması
    env = PendulumEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Ağırlık matrisi
    Q = np.array([
        [2, 0, 0, 0],       # Yatay konum
        [0, 1, 0, 0],       # Yatay hız
        [0, 0, 5, 0],       # Açısal pozisyon
        [0, 0, 0, 1]        # Açısal hız
    ])
    
    # Veri toplama
    states_history = []
    reward_states_history = []
    rewards_history = []
    
    for e in range(episodes):
        state = env.initial_state
        state = np.reshape(state, [1, 4])
        print(f"Episode {e+1}/{episodes}: Initial State: {state}")
        total_reward = 0
        
        if POISSON_IMPATCS:
            num_impacts = np.random.poisson(poisson_lambda)  # Darbe sayısı
            impact_steps = np.sort(np.random.choice(range(max_steps), size=num_impacts, replace=False))  # Darbe adımları
            impact_forces = np.random.uniform(-10, 10, size=num_impacts)  # Darbe kuvvetleri
            
            print(f"Episode {e+1}/{episodes}:")
            print(f"  Darbe Adımları: {impact_steps}")
            print(f"  Darbe Kuvvetleri: {impact_forces}")


        for step in range(max_steps):
            # Aksiyon seç ve uygula
            action = agent.act(state)
            force = env.force_values[action]

            if POISSON_IMPATCS:
                if step in impact_steps:
                    impact_index = np.where(impact_steps == step)[0][0]  # Hangi darbe olduğunu bul
                    force += impact_forces[impact_index]
                    print(f"  Adım {step}: Darbe Uygulandı! Kuvvet: {impact_forces[impact_index]:.2f}")
            
            # Çevreyi güncelle
            next_state = env.step(state[0], force)
            next_state = np.reshape(next_state, [1, 4])

            x = next_state[0, 0]
            xdot = next_state[0, 1]
            theta = (np.pi - next_state[0, 2] + np.pi) % (2 * np.pi) - np.pi    # [-π, π] aralığında normalize et
            thetadot = next_state[0, 3]
            print (f"  Adım {step+1}: x: {x:.2f}, xdot: {xdot:.2f}, theta: {theta:.2f}, thetadot: {thetadot:.2f}, F: {force:.2f}")

            # Ödül hesapla
            reward_state = np.array([x, xdot, theta, thetadot])
            reward_state = np.reshape(reward_state, [1, 4])
            reward = -(0.1 * np.dot(np.dot(reward_state, Q), reward_state.T).item() + 0.01 * (force**2))
            # reward = -(np.cos(theta)) - (abs(x) > 5)
            # # reward = -( np.dot(np.dot(reward_state, Q), np.array(reward_state).T).item() + 0.1*abs(x) + 0.1*abs(theta) )
            # reward = -np.dot(np.dot(state, Q), state.T).item()
            
            # Bölüm bitti mi kontrol et, bittiyse cezalandır.
            done = (abs(theta) >= np.pi/4) or (abs(x) > 5)
            if done: 
                reward -= 0.05 * (max_steps - step)
                print(f"  Adım {step+1}: Bölüm bitirildi! Theta ve x: {theta:.2f}, {x:.2f}")
            

            # Hafızaya ekle ve öğren
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            states_history.append(state)
            reward_states_history.append(reward_state)
            
            # Replay ve model güncelleme
            if step % 2 :
                agent.replay()
                agent.update_target_model()

            if done or step == max_steps-1:
                print(f"episode: {e+1}/{episodes}, score: {total_reward:.6f}, epsilon: {agent.epsilon:.6f}")
                rewards_history.append(total_reward)
                break

    
    return agent, np.array(states_history), np.array(rewards_history), np.array(reward_states_history)   

if __name__ == "__main__":
    # Eğitimi çalıştır
    agent, states, rewards, reward_states = train(episodes=5)
    
    # Sonuçları kaydet
    agent.save_agent(f"{save_folder}")
    np.save(f"{save_folder}/states.npy", states)
    np.save(f"{save_folder}/reward_states.npy", reward_states)
    np.save(f"{save_folder}/rewards.npy", rewards)
    print("Eğitim tamamlandı. Durum ve ödül geçmişi kaydedildi.")