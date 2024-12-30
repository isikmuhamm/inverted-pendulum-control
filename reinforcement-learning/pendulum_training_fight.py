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
POISSON_IMPACTS = True
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
            Dense(self.action_size * 2, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size), random.randrange(self.action_size)]
        act_values = self.model.predict(state, verbose=0)
        balance_values = act_values[0][:self.action_size]
        attack_values = act_values[0][self.action_size:]
        return [np.argmax(balance_values), np.argmax(attack_values)]

    def remember(self, states, actions, rewards, next_states, done):
        self.memory.append((states, actions, rewards, next_states, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        next_q_values = self.model.predict(next_states, verbose=0)
        target_q_values = self.target_model.predict(next_states, verbose=0)
        current_q_values = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            # Dengeleme aksiyonu
            balance_q = next_q_values[i][:self.action_size]
            balance_next_action = np.argmax(balance_q)
            current_q_values[i][actions[i][0]] = rewards[i] + \
                self.gamma * target_q_values[i][balance_next_action] * (1 - dones[i])
            
            # Saldırı aksiyonu
            attack_q = next_q_values[i][self.action_size:]
            attack_next_action = np.argmax(attack_q)
            current_q_values[i][self.action_size + actions[i][1]] = rewards[i] + \
                self.gamma * target_q_values[i][self.action_size + attack_next_action] * (1 - dones[i])

        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
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
    env1 = PendulumEnvironment()
    env2 = PendulumEnvironment()
    agent = DQNAgent(env1.state_size * 2, env1.action_size)

    Q = np.array([
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 1]
    ])

    # Veri toplama
    states_L_history = []
    states_R_history = []
    reward_states_L_history = []
    reward_states_R_history = []
    rewards_L_history = []
    rewards_R_history = []

    for e in range(episodes):
        state_L = np.reshape(env1.initial_state, [1, 4])
        state_L[0][0] -= +2.00
        state_R = np.reshape(env2.initial_state, [1, 4])
        state_R[0][0] += +2.00
        state_LL = np.hstack((state_L, state_R))
        state_RR = np.hstack((state_R, state_L))
        print(f"Episode {e+1}/{episodes}: Initial State ---> Left Cart: {state_L}, Right Cart: {state_R}")
        total_reward_L = 0
        total_reward_R = 0

        if POISSON_IMPACTS:
            poisson_steps_L = np.sort(np.random.choice(range(max_steps), size=np.random.poisson(poisson_lambda), replace=False))
            print(f"  Poisson Darbe Adımları (L) {len(poisson_steps_L)} adım: {poisson_steps_L}")
            poisson_steps_R = np.sort(np.random.choice(range(max_steps), size=np.random.poisson(poisson_lambda), replace=False))
            print(f"  Poisson Darbe Adımları (R) {len(poisson_steps_R)} adım: {poisson_steps_R}")

        for step in range(max_steps):

            # Durumlara göre denge ve saldırı kuvvetlerini belirle
            actions_L = agent.act(state_LL)
            force_L_balance = env1.force_values[actions_L[0]]
            force_L_attack = env1.force_values[actions_L[1]]
            print(f"  Adım {step+1}: L tarafından görülen durum ve alınan aksiyon:{state_LL} ---> {actions_L} Buna göre denge: {force_L_balance:.2f}, saldırı: {force_L_attack:.2f}")

            actions_R = agent.act(state_RR)
            force_R_balance = env2.force_values[actions_R[0]]
            force_R_attack = env2.force_values[actions_R[1]]
            print(f"  Adım {step+1}: L tarafından görülen durum ve alınan aksiyon: {state_RR} ---> {actions_R} Buna göre denge: {force_R_balance:.2f}, saldırı: {force_R_attack:.2f}")

            # Darbe uygula ve uygulanan darbeyi paylaş
            if POISSON_IMPACTS and step in poisson_steps_L:
                force_R_balance += force_L_attack
                print(f"  Adım {step+1}: L tarafından belirlenen darbe adımında darbe kuvveti uygulandı!")
       
            if POISSON_IMPACTS and step in poisson_steps_R:
                force_L_balance += force_R_attack
                print(f"  Adım {step+1}: R tarafından belirlenen darbe adımında darbe kuvveti uygulandı!")

            # Çevreyi güncelle
            next_state_L = np.reshape(env1.step(state_L[0], force_L_balance), [1, 4])
            next_state_R = np.reshape(env2.step(state_R[0], force_R_balance), [1, 4])
            next_state_LL = np.hstack((next_state_L, next_state_R))
            next_state_RR = np.hstack((next_state_R, next_state_L))

            # Dinamik modelin hedef açıyla olan farkını normalize et
            x_L = next_state_L[0, 0]
            xdot_L = next_state_L[0, 1]
            theta_L = (np.pi - next_state_L[0, 2] + np.pi) % (2 * np.pi) - np.pi    # [-π, π] aralığında normalize et
            thetadot_L = next_state_L[0, 3]
            print (f"L aracı Adım {step+1}: x: {x_L:.2f}, xdot: {xdot_L:.2f}, theta: {theta_L:.2f}, thetadot: {thetadot_L:.2f}, Toplam F: {force_L_balance:.2f}")

            x_R = next_state_R[0, 0]
            xdot_R = next_state_R[0, 1]
            theta_R = (np.pi - next_state_R[0, 2] + np.pi) % (2 * np.pi) - np.pi    # [-π, π] aralığında normalize et
            thetadot_R = next_state_R[0, 3]
            print (f"R aracı Adım {step+1}: x: {x_R:.2f}, xdot: {xdot_R:.2f}, theta: {theta_R:.2f}, thetadot: {thetadot_R:.2f}, Toplam F: {force_R_balance:.2f}")

            # Ödül hesapla
            reward_state_L = np.array([x_L, xdot_L, theta_L, thetadot_L])
            reward_state_L = np.reshape(reward_state_L, [1, 4])
            reward_state_R = np.array([x_R, xdot_R, theta_R, thetadot_R])
            reward_state_R = np.reshape(reward_state_R, [1, 4])

            reward_L = -(0.1 * np.dot(np.dot(reward_state_L, Q), reward_state_L.T).item() + 0.01 * (force_L_balance**2))
            reward_R = -(0.1 * np.dot(np.dot(reward_state_R, Q), reward_state_R.T).item() + 0.01 * (force_R_balance**2))

            reward_L_final = reward_L - reward_R
            reward_R_final = reward_R - reward_L

            # Bölüm bitti mi kontrol et, bittiyse cezalandır.
            done_L = (abs(theta_L) >= np.pi / 4 or abs(x_L) > 10)
            if done_L: 
                reward_L_final -= 0.05 * (max_steps - step)
                print(f"  L tarafından bölüm bitti. L bölümü kaybetti.")

            done_R = (abs(theta_R) >= np.pi / 4 or abs(x_R) > 10)
            if done_R: 
                reward_R_final -= 0.05 * (max_steps - step)
                print(f"  R tarafından bölüm bitti. R bölümü kaybetti.")

            done = done_L or done_R

            # Hafızaya ekle ve öğren
            agent.remember(state_LL, actions_L, reward_L_final, next_state_LL, done_L)
            agent.remember(state_RR, actions_R, reward_R_final, next_state_RR, done_R)

            # Durumu güncelle ve bilgileri topla
            state_L = next_state_L
            states_L_history.append(state_L)
            reward_states_L_history.append(reward_state_L)
            state_R = next_state_R
            states_R_history.append(state_R)
            reward_states_R_history.append(reward_state_R)
            total_reward_L += reward_L_final
            total_reward_R += reward_R_final

            # Replay ve model güncelleme
            if step % 3 == 0:
                agent.replay()
                agent.update_target_model()

            if done or step == max_steps - 1:
                print(f"Episode: {e+1}/{episodes}, Left Reward: {total_reward_L:.2f}, Right Reward: {total_reward_R:.2f}, Epsilon: {agent.epsilon:.5f}")
                rewards_L_history.append(total_reward_L)
                rewards_R_history.append(total_reward_R)
                break

    return agent, np.array(states_L_history), np.array(states_R_history), np.array(rewards_L_history), np.array(rewards_R_history), np.array(reward_states_L_history), np.array(reward_states_R_history)

if __name__ == "__main__":

    # Eğitimi çalıştır
    agent, states_L, states_R, rewards_L, rewards_R, reward_states_L, reward_states_R = train(episodes=5)

    # Sonuçları kaydet
    agent.save_agent(f"{save_folder}")
    np.save(f"{save_folder}/states_L.npy", states_L)
    np.save(f"{save_folder}/states_R.npy", states_R)
    np.save(f"{save_folder}/rewards_L.npy", rewards_L)
    np.save(f"{save_folder}/rewards_R.npy", rewards_R)
    np.save(f"{save_folder}/reward_states_L.npy", reward_states_L)
    np.save(f"{save_folder}/reward_states_R.npy", reward_states_R)
    print("Eğitim tamamlandı ve sonuçlar kaydedildi.")