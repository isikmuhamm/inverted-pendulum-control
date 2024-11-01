import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pendulum_nonlinear_model import PendulumEnvironment
import os

class PendulumVisualizer:
    def __init__(self):
        self.env = PendulumEnvironment()
        self.l = self.env.l
        
    def create_cart_pendulum_animation(self, states, interval=50):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True)

        # Araba ve sarkacın çizimi
        cart, = ax.plot([], [], 'ks-', lw=10)  # Aracı temsil eden siyah bir kare
        pendulum_line, = ax.plot([], [], 'ro-', lw=2, label='Pendulum')  # Sarkaç çizgisi
        trace, = ax.plot([], [], ':b', alpha=0.3, label='Pendulum Path') #Sarkacın izlediği yol

        # Metin alanları için
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        angle_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        angledot_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
        x_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        xdot_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        
        def init():
            cart.set_data([], [])
            pendulum_line.set_data([], [])
            time_text.set_text('')
            angle_text.set_text('')
            angledot_text.set_text('')
            x_text.set_text('')
            xdot_text.set_text('')
            return cart, pendulum_line, time_text, angle_text, angledot_text, x_text, xdot_text

        def animate(i):
            x = states[i, 0]  # Arabanın pozisyonu
            theta = states[i, 2]  # Sarkacın açısı (radyan cinsinden)
            xdot = states[i, 1]  # Arabanın hızı
            theta_dot = states[i, 3]  # Sarkacın açısı hızı
            
            # Aracın pozisyonunu güncelle
            cart_x = [x - 0.1, x + 0.1]  # Aracı bir kare olarak modelle
            cart_y = [0, 0]
            cart.set_data(cart_x, cart_y)
            
            # Sarkaç ucu pozisyonu
            pendulum_x = [x, x + self.l * np.sin(theta)]  # Sarkaç yatay pozisyonu
            pendulum_y = [0, self.l * np.cos(theta)]       # Sarkaç dikey pozisyonu
            pendulum_line.set_data(pendulum_x, pendulum_y)

            # Metin bilgilerini güncelle
            time_text.set_text(f'Time: {i * 0.02:.2f} [s]')
            angle_text.set_text(f'Pendulum Angle: {theta:.2f} [rad]')
            angledot_text.set_text(f'Pendulum Angular Speed: {theta_dot:.2f} [rad/s]')
            x_text.set_text(f'Cart Location: {x:.2f} [m]')
            xdot_text.set_text(f'Cart Speed: {xdot:.2f} [m/s]')

            return cart, pendulum_line, time_text, angle_text, x_text, xdot_text, angledot_text

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(states), interval=interval, blit=True)

        plt.title('Cart-Pendulum System')
        plt.legend(loc='upper right')
        plt.show()

    def plot_states(self, states, title_prefix=""):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        t = np.arange(len(states)) * self.env.time_step
        
        variables = ['Cart Position (m)', 'Cart Velocity (m/s)', 
                    'Pendulum Angle (rad)', 'Angular Velocity (rad/s)']
        
        colors = ['b', 'g', 'r', 'm']
        
        for i, (ax, var) in enumerate(zip(axs, variables)):
            ax.plot(t, states[:, i], color=colors[i])
            ax.set_title(f"{var} ({title_prefix})")
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
  
    def plot_phase_portraits(self, states):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        
        ax1.plot(states[:, 0], states[:, 1], 'b-', alpha=0.6)
        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Cart Phase Portrait')
        ax1.grid(True)
        
        ax2.plot(states[:, 2], states[:, 3], 'r-', alpha=0.6)
        ax2.set_xlabel('Angle (rad)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Pendulum Phase Portrait')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


    def plot_rewards(self, rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(rewards)
        plt.title('Training Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.show()

    def run_trained_agent(self, model_path):
        from tensorflow import keras
        try:
            # Eğitilmiş modeli yükle
            model = keras.models.load_model(model_path)
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            return

        env = PendulumEnvironment()
        action_size = 17
        force_values = np.linspace(-20, 20, action_size)  # Ajanın eğitimde kullandığı kuvvet değerleri

        initial_state = np.array([0.0, 0.0, np.random.uniform(-np.pi / 18, +np.pi / 18), 0.0])
        print(f"Başlangıç durumu: {initial_state}")
        state = initial_state.reshape(1, 4)  # Model için reshape
        states = [initial_state]  # İlk durumu listeye ekle

        for _ in range(500):  # 10 saniye simülasyon (50 Hz)
            # Modelin tahmin ettiği Q değerlerinden eylemi seç
            q_values = model.predict(state, verbose=0)[0]
            action_index = np.argmax(q_values)  # En yüksek Q-değerine sahip indeks
            action = force_values[action_index]  # İlgili kuvvet değeri
            
            print(f"Q-values: {q_values}")
            print(f"Seçilen Action (Kuvvet): {action}")

            next_state = np.array(env.step(states[-1], action))  # Son state'i kullanarak env'de bir adım at
            print(f"Next State: {next_state}")

            states.append(next_state)  # Yeni durumu listeye ekle
            state = next_state.reshape(1, 4)  # Bir sonraki tahmin için reshape

        # Animasyon ve grafikler oluştur
        states = np.array(states)
        self.create_cart_pendulum_animation(states)
        self.plot_states(states, "Trained Agent")
        self.plot_phase_portraits(states)
        
def main():
    visualizer = PendulumVisualizer()
    base_path = "reinforcement-learning"

    while True:
        print("\nTers Sarkaç Simülasyon Menüsü:")
        print("1 - Kuvvetsiz simülasyon animasyonu")
        print("2 - Kuvvetsiz simülasyon grafikleri")
        print("3 - Eğitim süreci animasyonu")
        print("4 - Eğitim süreci state grafikleri")
        print("5 - Eğitim süreci ödül grafikleri")
        print("6 - Eğitilmiş ajanın canlı simülasyonu")
        print("0 - Çıkış")
        
        try:
            choice = input("\nSeçiminiz: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = '0'
            print("0\n")

        if choice == '0':
            print("Program sonlandırılıyor...")
            break

        try:
            if choice == '1':
                file_path = os.path.join(base_path, "states_zero.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.create_cart_pendulum_animation(states)

            elif choice == '2':
                file_path = os.path.join(base_path, "states_zero.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.plot_states(states, "Free Simulation")

            elif choice == '3':
                file_path = os.path.join(base_path, "states.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.create_cart_pendulum_animation(states)

            elif choice == '4':
                file_path = os.path.join(base_path, "states.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.plot_states(states, "Training")
                visualizer.plot_phase_portraits(states)

            elif choice == '5':
                file_path = os.path.join(base_path, "rewards.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                rewards = np.load(file_path)
                visualizer.plot_rewards(rewards)

            elif choice == '6':
                file_path = os.path.join(base_path, "pendulum_model.keras")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                visualizer.run_trained_agent(file_path)

            else:
                print("Geçersiz seçim! Lütfen 0-6 arasında bir sayı giriniz.")

        except Exception as e:
            print(f"İşlem sırasında bir hata oluştu: {e}")
            continue

if __name__ == "__main__":
    main()