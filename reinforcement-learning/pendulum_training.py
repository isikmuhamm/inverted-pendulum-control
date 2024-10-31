import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from pendulum_nonlinear_model import PendulumEnvironment

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Model parametreleri
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 64
        self.target_update_counter = 0
        
        # Modeller
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter += 1

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

        targets = rewards + self.gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(self.batch_size), actions] = targets
        
        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=self.batch_size)
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_agent(self, folder_path):
        """Ajanın tüm durumunu kaydet"""
        # Model
        self.model.save(f'{folder_path}/model', save_format='tf')
        
        # Agent durumu
        agent_state = {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'memory': list(self.memory)
        }
        np.save(f'{folder_path}/agent_state.npy', agent_state)

    def load_agent(self, folder_path):
        """Ajanın tüm durumunu yükle"""
        # Model
        self.model = tf.keras.models.load_model(f'{folder_path}/model')
        self.target_model = tf.keras.models.load_model(f'{folder_path}/model')
        
        # Agent durumu
        agent_state = np.load(f'{folder_path}/agent_state.npy', allow_pickle=True).item()
        self.epsilon = agent_state['epsilon']
        self.learning_rate = agent_state['learning_rate']
        self.memory = deque(agent_state['memory'], maxlen=2000)

def train(episodes=1000, max_steps=200):
    # Eğitim parametreleri
    env = PendulumEnvironment()
    state_size = 4
    action_size = 17
    force_values = np.linspace(-20, 20, action_size)
    
    # Ajan oluştur
    agent = DQNAgent(state_size, action_size)
    
    # Ağırlık matrisi
    Q = np.array([
        [1, 0, 0, 0],      # Yatay konum
        [0, 1, 0, 0],      # Yatay hız
        [0, 0, 100, 0],    # Açısal pozisyon
        [0, 0, 0, 10]      # Açısal hız
    ])
    
    # Veri toplama
    states_history = []
    rewards_history = []
    
    for e in range(episodes):
        state = np.array([0.0, 0.0, np.random.uniform(np.pi-np.pi/18, np.pi+np.pi/18), 0.0])
        state = np.reshape(state, [1, 4])
        total_reward = 0
        
        for step in range(max_steps):
            # Aksiyon seç ve uygula
            action = agent.act(state)
            force = force_values[action]
            next_state = env.step(state[0], force)
            next_state = np.reshape(next_state, [1, 4])
            reward_state = np.array([next_state[0, 0], next_state[0, 1], np.pi - next_state[0, 2], next_state[0, 3]])

            # Ödül hesapla
            reward = -np.dot(np.dot(reward_state, Q), reward_state.T).item()
            
            # Bölüm bitti mi kontrol et
            theta = next_state[0, 2]
            done = (abs(np.pi-theta) >= np.pi/4)
            
            # Hafızaya ekle ve öğren
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            states_history.append(state[0])
            
            if done or step == max_steps-1:
                print(f"episode: {e}/{episodes}, score: {total_reward:.5f}, epsilon: {agent.epsilon:.5f}")
                rewards_history.append(total_reward)
                break
        
        agent.replay()
        
        # Periyodik güncellemeler
        if e and e % 10 == 0:
            agent.update_target_model()
        if e and e % 100 == 0:
            agent.learning_rate *= 0.5
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            print(f"%%% Parametre güncellemesi yapıldı. Learning rate: {agent.learning_rate:.5f}, Epsilon: {agent.epsilon:.5f}. %%%")
    
    return agent, np.array(states_history), np.array(rewards_history)

if __name__ == "__main__":
    # Eğitimi çalıştır
    agent, states, rewards = train()
    
    # Sonuçları kaydet
    save_folder = "reinforcement-learning"
    agent.save_agent(f"{save_folder}")
    np.save(f"{save_folder}/states.npy", states)
    np.save(f"{save_folder}/rewards.npy", rewards)
    
    print("Eğitim tamamlandı. Sonuçlar kaydedildi.")