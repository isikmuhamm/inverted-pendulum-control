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
from logger import Logger, load_and_save

CONTINUE_TRAINING = True
POISSON_IMPACTS = False
FIGHT_STATE = False
DOUBLE_MODE = False
TEST_MODE = False
if FIGHT_STATE: DOUBLE_MODE = True
if TEST_MODE: CONTINUE_TRAINING = True

save_folder = "reinforcement-learning"
logger = Logger("training.log")
DTYPE = np.float32
NO_ATTACK_INDEX = 4
max_len = 100000
poisson_lambda = 10

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Model parametreleri
        self.state_size = state_size
        self.action_size = action_size
        self.half_actions = action_size // 2
        self.memory = deque(maxlen=max_len)
        self.gamma = 0.99
        self.epsilon = 1.00
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.minimum_memory = 5000
        
        # Eğitime kaldığın yerden devam et
        if CONTINUE_TRAINING:
            self.load_agent(f"{save_folder}")
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        act_values = self.model.predict(state, verbose=0)
        balance_values = act_values[0][:self.half_actions]
        attack_values = act_values[0][self.half_actions:]
        if np.random.rand() <= self.epsilon:
            # Keşif
            balance_idx = np.argmax(attack_values) if FIGHT_STATE and CONTINUE_TRAINING else random.randrange(self.half_actions)
            attack_idx = random.randrange(self.half_actions) if FIGHT_STATE else NO_ATTACK_INDEX
        else:
            # En iyi bilinen değerleri kullan
            balance_idx = np.argmax(balance_values)
            attack_idx = np.argmax(attack_values) if FIGHT_STATE else NO_ATTACK_INDEX
        
        return [balance_idx, attack_idx]

    def remember(self, states, actions, rewards, next_states, done):
        states = states.astype(DTYPE)
        rewards = np.array(rewards, dtype=DTYPE)
        next_states = next_states.astype(DTYPE)
        self.memory.append((states, actions, rewards, next_states, done))

    def replay(self):
        if len(self.memory) < self.minimum_memory:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0][0] for x in minibatch], dtype=DTYPE)
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch], dtype=DTYPE)
        next_states = np.array([x[3][0] for x in minibatch], dtype=DTYPE)
        dones = np.array([x[4] for x in minibatch])

        next_q_values = self.model.predict(next_states, verbose=0)
        target_q_values = self.target_model.predict(next_states, verbose=0)
        current_q_values = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            # Balance aksiyonu için Q-değeri güncelleme
            balance_q = next_q_values[i][:self.half_actions]
            balance_next_action = np.argmax(balance_q)
            balance_target = rewards[i] + \
                self.gamma * target_q_values[i][balance_next_action] * (1 - dones[i])
            current_q_values[i][actions[i][0]] = balance_target
            
            # Saldırı aksiyonu için Q-değeri güncelleme (sadece savaş durumunda)
            if FIGHT_STATE:
                attack_q = next_q_values[i][self.half_actions:]
                attack_next_action = np.argmax(attack_q)
                attack_target = rewards[i] + \
                    self.gamma * target_q_values[i][self.half_actions + attack_next_action] * (1 - dones[i])
                current_q_values[i][self.half_actions + actions[i][1]] = attack_target

        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_agent(self, folder_path):
        """Ajanın tüm durumunu kaydet"""

        if TEST_MODE:
            print("Test modu açık olduğu için ajan durumu kaydedilmedi.")
            return

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
            if FIGHT_STATE and (self.epsilon == 0.01): self.epsilon = 1.00 # Dövüş modu için exploration'ı sıfırla
            self.learning_rate = agent_state.get('learning_rate', self.learning_rate)
            self.memory = deque(agent_state.get('memory', []), maxlen=max_len)
            print(f"Ajan durumu başarıyla {agent_state_path} konumundan yüklendi.")
        
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            print("Yeni model oluşturuluyor...")
            self.model = self._build_model()
            self.target_model = self._build_model()

def train(episodes=2000, max_steps=200):
    env1 = PendulumEnvironment()
    if DOUBLE_MODE: env2 = PendulumEnvironment()
    agent = DQNAgent( (env1.state_size * 2) + 1, env1.action_size * 2)

    Q = np.array([
        [2, 0, 00, 0],
        [0, 1, 00, 0],
        [0, 0, 10, 0],
        [0, 0, 00, 4]
    ], dtype=DTYPE)

    # Veri toplama
    states_L_history, states_R_history = [], []
    reward_states_L_history, reward_states_R_history = [], []
    rewards_L_history, rewards_R_history = [], []
    steps_per_episode = []

    for e in range(episodes):
        state_L = np.reshape(env1.initial_state, [1, 4])
        if DOUBLE_MODE: state_R = np.reshape(env2.initial_state, [1, 4])

        external_impact_L = 0
        external_impact_R = 0

        if FIGHT_STATE:
            state_LL = np.hstack((state_L, state_R, [[external_impact_L]]))
            state_RR = np.hstack((state_R, state_L, [[external_impact_R]]))
        else:
            state_LL = np.hstack((state_L, [[0,0,0,0]], [[external_impact_L]]))
            if DOUBLE_MODE: state_RR = np.hstack((state_R, [[0,0,0,0]], [[external_impact_R]]))

        if DOUBLE_MODE: print(f"Episode {e+1}/{episodes}: Initial State ---> Left Cart: {state_L}, Right Cart: {state_R}")
        else: print(f"Episode {e+1}/{episodes}: Initial State ---> Left Cart: {state_L}")
        if DOUBLE_MODE: logger.log("train", f"Episode {e+1}/{episodes}: Initial State ---> Left Cart: {state_L}, Right Cart: {state_R}")
        else: logger.log("train", f"Episode {e+1}/{episodes}: Initial State ---> Left Cart: {state_L}")

        total_reward_L, total_reward_R = 0, 0

        if POISSON_IMPACTS:
            poisson_steps = np.sort(np.random.choice(range(max_steps), size=np.random.poisson(poisson_lambda), replace=False))
            print(f"  Poisson Darbe Adımları {len(poisson_steps)} adım: {poisson_steps}")
            logger.log("train", f"  Poisson Darbe Adımları {len(poisson_steps)} adım: {poisson_steps}")
            impact_forces = np.random.uniform(-10, 10, size=poisson_steps)


        for step in range(max_steps):

            # Durumlara göre denge ve saldırı kuvvetlerini belirle
            actions_L = agent.act(state_LL)
            force_L_balance = env1.force_values[actions_L[0]]
            force_L_attack = env1.attack_values[actions_L[1]]

            if DOUBLE_MODE:
                actions_R = agent.act(state_RR)
                force_R_balance = env2.force_values[actions_R[0]]
                force_R_attack = env2.attack_values[actions_R[1]]
            else: force_R_balance, force_R_attack = 0,0

            if DOUBLE_MODE: print(f"  Adım {step+1}: L aksiyon: {actions_L} Denge: {force_L_balance:+06.2f}, saldırı: {force_L_attack:+06.2f}\t\t\t\tR aksiyon: {actions_R} Denge: {force_R_balance:+06.2f}, saldırı: {force_R_attack:+06.2f}")
            else: print(f"  Adım {step+1}: L aksiyon: {actions_L} Denge: {force_L_balance:+06.2f}, saldırı: {force_L_attack:+06.2f}")
            if DOUBLE_MODE: logger.log("train", f"  Adım {step+1}: L aksiyon: {actions_L} Denge: {force_L_balance:+06.2f}, saldırı: {force_L_attack:+06.2f}\t\t\t\tR aksiyon: {actions_R} Denge: {force_R_balance:+06.2f}, saldırı: {force_R_attack:+06.2f}")
            else: logger.log("train", f"  Adım {step+1}: L aksiyon: {actions_L} Denge: {force_L_balance:+06.2f}, saldırı: {force_L_attack:+06.2f}")

            # Rastgele bozucu darbe uygula ve uygulanan darbeyi paylaş
            if POISSON_IMPACTS:
                if step in poisson_steps:
                    impact_index = np.where(poisson_steps == step)[0][0]  # Hangi darbe olduğunu bul
                    poisson_force = impact_forces[impact_index]
                    print(f"  Adım {step+1}: {impact_forces[impact_index]:.2f} kuvvetinde bir rastgele bozucu darbe uygulandı!")
                    logger.log("train", f"  Adım {step+1}: {impact_forces[impact_index]:.2f} kuvvetinde bir rastgele bozucu darbe uygulandı!")
                else: poisson_force = 0
            else: poisson_force = 0

            total_force_L = force_L_balance + force_R_attack + poisson_force
            total_force_R = force_R_balance + force_L_attack + poisson_force

            external_impact_L = force_R_attack + poisson_force
            external_impact_R = force_L_attack + poisson_force

            # Çevreyi güncelle
            next_state_L = np.reshape(env1.step(state_L[0], total_force_L), [1, 4])
            if DOUBLE_MODE: next_state_R = np.reshape(env2.step(state_R[0], total_force_R), [1, 4])
            
            if FIGHT_STATE:
                next_state_LL = np.hstack((next_state_L, next_state_R, [[external_impact_L]]))
                next_state_RR = np.hstack((next_state_R, next_state_L, [[external_impact_R]]))
            else:
                next_state_LL = np.hstack((next_state_L, [[0,0,0,0]], [[external_impact_L]]))
                if DOUBLE_MODE: next_state_RR = np.hstack((next_state_R, [[0,0,0,0]], [[external_impact_R]]))

            # Dinamik modelin hedef açıyla olan farkını normalize et
            x_L = next_state_L[0, 0]
            xdot_L = next_state_L[0, 1]
            theta_L = (np.pi - next_state_L[0, 2] + np.pi) % (2 * np.pi) - np.pi    # [-π, π] aralığında normalize et
            thetadot_L = next_state_L[0, 3]

            if DOUBLE_MODE:
                x_R = next_state_R[0, 0]
                xdot_R = next_state_R[0, 1]
                theta_R = (np.pi - next_state_R[0, 2] + np.pi) % (2 * np.pi) - np.pi    # [-π, π] aralığında normalize et
                thetadot_R = next_state_R[0, 3]

            # Ödül hesapla
            reward_state_L = np.array([x_L, xdot_L, theta_L, thetadot_L])
            reward_state_L = np.reshape(reward_state_L, [1, 4])
            
            if DOUBLE_MODE:
                reward_state_R = np.array([x_R, xdot_R, theta_R, thetadot_R])
                reward_state_R = np.reshape(reward_state_R, [1, 4])


            reward_L = -(0.1 * np.dot(np.dot(reward_state_L, Q), reward_state_L.T).item() + 0.001 * (force_L_balance**2) + 0.0005 * (force_L_attack**2) + 0.2 * (abs(theta_L) > np.pi/8) + 0.1 * (abs(theta_L) * abs(thetadot_L))  )
            if DOUBLE_MODE: reward_R = -(0.1 * np.dot(np.dot(reward_state_R, Q), reward_state_R.T).item() + 0.001 * (force_R_balance**2) + 0.0005 * (force_R_attack**2) + 0.2 * (abs(theta_R) > np.pi/8) + 0.1 * (abs(theta_R) * abs(thetadot_R))  )


            # Kavga modu açıkken rakibin dengesizlik durumuna göre ödül ayarlanır
            if FIGHT_STATE: reward_L_final = reward_L - 0.3*reward_R
            else: reward_L_final = reward_L
            if FIGHT_STATE: reward_R_final = reward_R - 0.3*reward_L
            elif DOUBLE_MODE: reward_R_final = reward_R


            # Bölüm bitti mi kontrol et, bittiyse cezalandır.
            done_L = (abs(theta_L) >= np.pi / 4 or abs(x_L) > 5)
            if done_L: 
                reward_L_final -= 0.1 * (max_steps - step)
                print(f"  L tarafından bölüm bitti. L bölümü kaybetti.")
                logger.log("train", f"  L tarafından bölüm bitti. L bölümü kaybetti.")

            if DOUBLE_MODE:
                done_R = (abs(theta_R) >= np.pi / 4 or abs(x_R) > 5)
                if done_R: 
                    reward_R_final -= 0.1 * (max_steps - step)
                    print(f"  R tarafından bölüm bitti. R bölümü kaybetti.")
                    logger.log("train", f"  R tarafından bölüm bitti. R bölümü kaybetti.")


            if DOUBLE_MODE: print (f"  Adım {step+1}: L durum: [{x_L:.2f}, {xdot_L:.2f}, {theta_L:.2f}, {thetadot_L:.2f}], Toplam F: {force_L_balance:+06.2f}, Ödül: {reward_L_final:+06.2f},\t\tR durum: [{x_R:.2f}, {xdot_R:.2f}, {theta_R:.2f}, {thetadot_R:.2f}], Toplam F: {force_R_balance:+06.2f}, Ödül: {reward_R_final:+06.2f}")
            else: print (f"  Adım {step+1}: L durum: [{x_L:.2f}, {xdot_L:.2f}, {theta_L:.2f}, {thetadot_L:.2f}], Toplam F: {force_L_balance:+06.2f}, Ödül: {reward_L_final:+06.2f}")
            if DOUBLE_MODE: logger.log("train", f"  Adım {step+1}: L durum: [{x_L:.2f}, {xdot_L:.2f}, {theta_L:.2f}, {thetadot_L:.2f}], Toplam F: {force_L_balance:+06.2f}, Ödül: {reward_L_final:+06.2f},\t\tR durum: [{x_R:.2f}, {xdot_R:.2f}, {theta_R:.2f}, {thetadot_R:.2f}], Toplam F: {force_R_balance:+06.2f}, Ödül: {reward_R_final:+06.2f}")
            else: logger.log("train", f"  Adım {step+1}: L durum: [{x_L:.2f}, {xdot_L:.2f}, {theta_L:.2f}, {thetadot_L:.2f}], Toplam F: {force_L_balance:+06.2f}, Ödül: {reward_L_final:+06.2f}")


            if DOUBLE_MODE: done = done_L or done_R
            else: done = done_L

            # Hafızaya ekle ve öğren
            agent.remember(state_LL, actions_L, reward_L_final, next_state_LL, done_L)
            if DOUBLE_MODE: agent.remember(state_RR, actions_R, reward_R_final, next_state_RR, done_R)

            # Durumu güncelle ve bilgileri topla
            state_L = next_state_L
            if DOUBLE_MODE: state_R = next_state_R
            states_L_history.append(state_L)
            reward_states_L_history.append(reward_state_L)
            if DOUBLE_MODE:
                states_R_history.append(state_R)
                reward_states_R_history.append(reward_state_R)

            total_reward_L += reward_L_final
            if DOUBLE_MODE: total_reward_R += reward_R_final
            
            # Replay ve model güncelleme
            if step % 1 == 0:
                agent.replay()
                agent.update_target_model()

            if done or step == max_steps - 1:
                if DOUBLE_MODE: print(f"Episode {e+1}/{episodes}: L Reward: {(total_reward_L/step):.2f}, R Reward: {(total_reward_R/step):.2f}, E: {agent.epsilon:.5f}")
                else: print(f"Episode {e+1}/{episodes}: L Reward: {(total_reward_L/step):.2f}, E: {agent.epsilon:.5f}")
                if DOUBLE_MODE: logger.log("train", f"Episode {e+1}/{episodes}: L Reward: {(total_reward_L/step):.2f}, R Reward: {(total_reward_R/step):.2f}, E: {agent.epsilon:.5f}")
                else: logger.log("train", f"Episode {e+1}/{episodes}: L Reward: {(total_reward_L/step):.2f}, E: {agent.epsilon:.5f}")


                rewards_L_history.append(total_reward_L)
                if DOUBLE_MODE: rewards_R_history.append(total_reward_R)
                steps_per_episode.append(step + 1)
                print("*"* 120)
                break

    return agent, np.array(states_L_history), np.array(states_R_history), np.array(rewards_L_history), np.array(rewards_R_history), np.array(reward_states_L_history), np.array(reward_states_R_history), np.array(steps_per_episode)


if __name__ == "__main__":

    print(f"Reinforcement learning ile iki sarkacın aynı ajan tarafından kontrol edilmesi eğitimine hoş geldiniz.")
    print(f"Kavga modu: {FIGHT_STATE}, Poisson Darbe: {POISSON_IMPACTS}, Eğitim devam durumu: {CONTINUE_TRAINING}")
    print("Eğitimi kavga modunda çalıştırmadan önce kavgasız ve possion darbe modunda çalıştırmanız ve ajanı bu")
    print("şekilde eğittikten sonra kavga modunu açmanız önerilir. Ctrl+C ile eğitimi istediğiniz zaman durdurabilirsiniz.")
    logger.log("train", f"Reinforcement learning ile iki sarkacın aynı ajan tarafından kontrol edilmesi eğitimine hoş geldiniz.")
    logger.log("train", f"Kavga modu: {FIGHT_STATE}, Poisson Darbe: {POISSON_IMPACTS}, Eğitim devam durumu: {CONTINUE_TRAINING}")
    logger.log("train", "Eğitimi kavga modunda çalıştırmadan önce kavgasız ve possion darbe modunda çalıştırmanız ve ajanı bu")
    logger.log("train", "şekilde eğittikten sonra kavga modunu açmanız önerilir. Ctrl+C ile eğitimi istediğiniz zaman durdurabilirsiniz.")
    
    batch_size = 1000  # Her bir batch'teki episode sayısı
    batch_number = 0
    
    while True:
        try:
            print(f"\nBatch {batch_number + 1} başlıyor...")
            logger.log("train", f"\nBatch {batch_number + 1} başlıyor...")
            
            # Batch training
            agent, states_L, states_R, rewards_L, rewards_R, reward_states_L, reward_states_R, steps_per_episode = train(episodes=batch_size)
            
            # Save agent first
            agent.save_agent(f"{save_folder}")
            
            # Save data
            load_and_save(f"{save_folder}/states_L.npy", np.array(states_L, dtype=DTYPE))
            load_and_save(f"{save_folder}/states_R.npy", np.array(states_R, dtype=DTYPE))
            load_and_save(f"{save_folder}/rewards_L.npy", np.array(rewards_L, dtype=DTYPE))
            load_and_save(f"{save_folder}/rewards_R.npy", np.array(rewards_R, dtype=DTYPE))
            load_and_save(f"{save_folder}/reward_states_L.npy", np.array(reward_states_L, dtype=DTYPE))
            load_and_save(f"{save_folder}/reward_states_R.npy", np.array(reward_states_R, dtype=DTYPE))
            load_and_save(f"{save_folder}/steps_per_episode.npy", np.array(steps_per_episode, dtype=DTYPE))
            

            # Batch raporu
            print(f"Batch {batch_number + 1} tamamlandı ---> Epsilon değeri: {agent.epsilon:.5f}, Ort Episode Uzunluğu: {np.mean(steps_per_episode):.2f}, Ort L ödül: {np.mean(rewards_L):.2f}, Ort R ödül: {np.mean(rewards_R):.2f}")
            print(f"Toplam durum: {'L kazandı' if sum(rewards_L) > sum(rewards_R) else 'R kazandı' if sum(rewards_R) > sum(rewards_L) else 'Berabere'}")
            print("*" * 120)
            logger.log("train", f"Batch {batch_number + 1} tamamlandı ---> Epsilon değeri: {agent.epsilon:.5f}, Ort Episode Uzunluğu: {np.mean(steps_per_episode):.2f}, Ort L ödül: {np.mean(rewards_L):.2f}, Ort R ödül: {np.mean(rewards_R):.2f}")
            logger.log("train", f"Toplam durum: {'L kazandı' if sum(rewards_L) > sum(rewards_R) else 'R kazandı' if sum(rewards_R) > sum(rewards_L) else 'Berabere'}")
            logger.log("train", "*" * 120)
            
            # Clear lists after saving
            del states_L, states_R, rewards_L, rewards_R, reward_states_L, reward_states_R

            batch_number += 1
            
        except Exception as e:
            print(f"Error: {e}")
            if 'agent' in locals():
                agent.save_agent(f"{save_folder}")
            break