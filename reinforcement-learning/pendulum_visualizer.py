import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pendulum_nonlinear_model import PendulumEnvironment
import os

STATE_COMPRESSION = 100
REWARD_COMPRESSION = 15

"""
Bu kod dosyası, eğitim sürecinde elde edilen sonuçları görselleştirmek için kullanılır. Seçenekler:
1 - Kuvvetsiz simülasyon animasyonu: Kuvvetsiz simülasyon sırasında durum değişimi animasyonunu gösterir.
2 - Kuvvetsiz simülasyon state grafikleri: Kuvvetsiz simülasyon sırasında durum değişimi grafiklerini gösterir.
3 - Eğitim süreci animasyonu: Eğitim sürecindeki durum değişimi animasyonunu gösterir.
4 - Eğitim süreci state grafikleri: Eğitim sürecindeki durum değişimini ve faz portrelerini gösterir.
5 - Eğitim süreci ödül ve faz grafikleri: Eğitim sürecindeki ödül değişimini gösterir.
6 - Çift eğitim süreci animasyonu: İki farklı eğitim sürecindeki durum değişimi animasyonunu gösterir.
7 - Çift eğitim süreci state grafikleri: İki farklı eğitim sürecindeki durum değişimini gösterir.
8 - Çift eğitim süreci ödül ve faz grafikleri: İki farklı eğitim sürecindeki ödül değişimini ve faz portrelerini gösterir.

This code file is used to visualize the results obtained during the training process. Options:
1 - Zero-force simulation animation: Shows the state change animation during the zero-force simulation.
2 - Zero-force simulation state graphs: Shows the state change graphs during the zero-force simulation.
3 - Training process animation: Shows the state change animation during the training process.
4 - Training process state graphs: Shows the state change and phase portraits during the training process.
5 - Training process reward and phase graphs: Shows the reward change during the training process.
6 - Dual training process animation: Shows the state change animation during two different training processes.
7 - Dual training process state graphs: Shows the state change during two different training processes.
8 - Dual training process reward and phase graphs: Shows the reward change and phase portraits during two different training processes.
"""

class PendulumVisualizer:
    def __init__(self):
        self.env = PendulumEnvironment()
        self.l = self.env.l
        
    def create_cart_pendulum_animation(self, states, interval=50):
        # Grafik sınırlarını ve boyutlarını ayarla
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True)

        # Araba ve sarkacın çizimi
        cart, = ax.plot([], [], 'ks-', lw=10)  # Aracı temsil eden siyah bir kare
        pendulum_line, = ax.plot([], [], 'ro-', lw=2, label='Pendulum')  # Sarkaç çizgisi
        trace, = ax.plot([], [], ':b', alpha=0.3, label='Pendulum Path') #Sarkacın izlediği yol
        
        # İz için hafıza
        history_x = []
        history_y = []

        # Metin alanları için
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        angle_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        angledot_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
        x_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        xdot_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        
        def init():
            cart.set_data([], [])
            pendulum_line.set_data([], [])
            trace.set_data([], [])
            time_text.set_text('')
            angle_text.set_text('')
            angledot_text.set_text('')
            x_text.set_text('')
            xdot_text.set_text('')
            return cart, pendulum_line, trace, time_text, angle_text, angledot_text, x_text, xdot_text

        def animate(i):
            x = states[i,0, 0]  # Arabanın pozisyonu
            theta = states[i,0, 2]  # Sarkacın açısı (radyan cinsinden)
            xdot = states[i,0, 1]  # Arabanın hızı
            theta_dot = states[i,0, 3]  # Sarkacın açısı hızı
            
            # Aracın pozisyonunu güncelle
            cart_x = [x - 0.1, x + 0.1]  # Aracı bir kare olarak modelle
            cart_y = [0, 0]
            cart.set_data(cart_x, cart_y)
            
            # Görünüm sınırlarını arabaya göre güncelle, iptal etmek için yoruma al
            # ax.set_xlim(x - 2, x + 2)  # Arabayı merkeze al ve ±2 birim göster

            # Sarkaç ucu pozisyonu
            pendulum_x = [x, x + self.l * np.sin(theta)]  # Sarkaç yatay pozisyonu
            pendulum_y = [0, -self.l * np.cos(theta)]       # Sarkaç dikey pozisyonu
            pendulum_line.set_data(pendulum_x, pendulum_y)

            # İzi güncelle (her 3 noktada bir kaydet - performans için)
            if i % 3 == 0:
                history_x.append(pendulum_x)
                history_y.append(pendulum_y)
            trace.set_data(history_x, history_y)

            # Metin bilgilerini güncelle
            time_text.set_text(f'Time: {i * 0.02:.2f} [s]')
            angle_text.set_text(f'Pendulum Angle: {theta:.2f} [rad]')
            angledot_text.set_text(f'Pendulum Angular Speed: {theta_dot:.2f} [rad/s]')
            x_text.set_text(f'Cart Location: {x:.2f} [m]')
            xdot_text.set_text(f'Cart Speed: {xdot:.2f} [m/s]')

            return cart, pendulum_line, trace, time_text, angle_text, x_text, xdot_text, angledot_text

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(states), interval=interval, blit=True)

        plt.title('Cart-Pendulum System')
        plt.legend(loc='upper right')
        plt.show()

    def create_dual_cart_pendulum_animation(self, states_L, states_R, interval=50):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.grid(True)

        # Sol ve sağ arabalar için çizim nesneleri
        cart_L, = ax.plot([], [], 'bs-', lw=10, label='Left Cart')
        pendulum_L, = ax.plot([], [], 'bo-', lw=2, label='Left Pendulum')
        trace_L, = ax.plot([], [], ':b', alpha=0.3, label='Left Path')
        
        cart_R, = ax.plot([], [], 'rs-', lw=10, label='Right Cart')
        pendulum_R, = ax.plot([], [], 'ro-', lw=2, label='Right Pendulum')
        trace_R, = ax.plot([], [], ':r', alpha=0.3, label='Right Path')

        # İz için hafıza
        history_x_L, history_y_L = [], []
        history_x_R, history_y_R = [], []

        time_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, horizontalalignment='center')

        # Sol araç için metin alanları
        L_angle_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='blue')
        L_angledot_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='blue')
        L_x_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, color='blue')
        L_xdot_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, color='blue')

        # Sağ araç için metin alanları
        R_angle_text = ax.text(0.75, 0.95, '', transform=ax.transAxes, color='red')
        R_angledot_text = ax.text(0.75, 0.90, '', transform=ax.transAxes, color='red')
        R_x_text = ax.text(0.75, 0.85, '', transform=ax.transAxes, color='red')
        R_xdot_text = ax.text(0.75, 0.80, '', transform=ax.transAxes, color='red')

        def init():
            cart_L.set_data([], [])
            pendulum_L.set_data([], [])
            trace_L.set_data([], [])
            cart_R.set_data([], [])
            pendulum_R.set_data([], [])
            trace_R.set_data([], [])
            return cart_L, pendulum_L, trace_L, cart_R, pendulum_R, trace_R

        def animate(i):
            # Sol araba
            x_L = states_L[i,0, 0]
            theta_L = states_L[i,0, 2]
            xdot_L = states_L[i,0, 1]
            theta_dot_L = states_L[i,0, 3]

            cart_L.set_data([x_L - 0.1, x_L + 0.1], [0, 0])
            pendulum_x_L = [x_L, x_L + self.l * np.sin(theta_L)]
            pendulum_y_L = [0, -self.l * np.cos(theta_L)]
            pendulum_L.set_data(pendulum_x_L, pendulum_y_L)

            # Sağ araba
            x_R = states_R[i,0, 0]
            theta_R = states_R[i,0, 2]
            xdot_R = states_R[i,0, 1]
            theta_dot_R = states_R[i,0, 3]

            cart_R.set_data([x_R - 0.1, x_R + 0.1], [0, 0])
            pendulum_x_R = [x_R, x_R + self.l * np.sin(theta_R)]
            pendulum_y_R = [0, -self.l * np.cos(theta_R)]
            pendulum_R.set_data(pendulum_x_R, pendulum_y_R)

            # İzleri güncelle
            if i % 3 == 0:
                history_x_L.append(pendulum_x_L)
                history_y_L.append(pendulum_y_L)
                history_x_R.append(pendulum_x_R)
                history_y_R.append(pendulum_y_R)
            trace_L.set_data(history_x_L, history_y_L)
            trace_R.set_data(history_x_R, history_y_R)

            # Metin bilgilerini güncelle
            time_text.set_text(f'Time: {i * 0.02:.2f} [s]')
            
            L_angle_text.set_text(f'L Angle: {theta_L:.2f} [rad]')
            L_angledot_text.set_text(f'L Angular Speed: {theta_dot_L:.2f} [rad/s]')
            L_x_text.set_text(f'L Position: {x_L:.2f} [m]')
            L_xdot_text.set_text(f'L Speed: {xdot_L:.2f} [m/s]')

            R_angle_text.set_text(f'R Angle: {theta_R:.2f} [rad]')
            R_angledot_text.set_text(f'R Angular Speed: {theta_dot_R:.2f} [rad/s]')
            R_x_text.set_text(f'R Position: {x_R:.2f} [m]')
            R_xdot_text.set_text(f'R Speed: {xdot_R:.2f} [m/s]')

            return (cart_L, pendulum_L, trace_L, cart_R, pendulum_R, trace_R, 
                    time_text, L_angle_text, L_angledot_text, L_x_text, L_xdot_text,
                    R_angle_text, R_angledot_text, R_x_text, R_xdot_text)

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(states_L), 
                            interval=interval, blit=True)

        plt.title('Dual Cart-Pendulum System')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

    def plot_states(self, states, title_prefix=""):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        t = np.arange(len(states)) * self.env.time_step
        
        variables = ['Cart Position (m)', 'Cart Velocity (m/s)', 
                    'Pendulum Angle (rad)', 'Angular Velocity (rad/s)']
        
        colors = ['b', 'g', 'r', 'm']
        
        for i, (ax, var) in enumerate(zip(axs, variables)):
            ax.plot(t, states[:,0, i], color=colors[i])
            ax.set_title(f"{var} ({title_prefix})")
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
  
    def plot_phase_and_rewards(self, states, rewards, title_prefix=""):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[7, 3])
        
        # Cart Phase Portrait (üst sol)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(states[:,0, 1], states[:,0, 0], 'b-')
        ax1.set_ylabel('Position (m)')
        ax1.set_xlabel('Velocity (m/s)')
        ax1.set_title(f'{title_prefix} Cart Phase Portrait')
        ax1.grid(True)
        
        # Pendulum Phase Portrait (üst sağ)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(states[:,0, 3], states[:,0, 2], 'r-')
        ax2.set_ylabel('Angle (rad)')
        ax2.set_xlabel('Angular Velocity (rad/s)')
        ax2.set_title(f'{title_prefix} Pendulum Phase Portrait')
        ax2.grid(True)
        
        # Rewards (alt)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(rewards, 'g-')
        ax3.set_title(f'{title_prefix} Training Rewards Over Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    def compress(self, array, compression_factor=10):
        compressed_array = array[::compression_factor]
        return compressed_array

def main():
    visualizer = PendulumVisualizer()
    base_path = "reinforcement-learning"

    while True:
        print("\nTers Sarkaç Simülasyon Menüsü:")
        print("1 - Kuvvetsiz simülasyon animasyonu")
        print("2 - Kuvvetsiz simülasyon grafikleri")
        print("3 - Eğitim süreci animasyonu")
        print("4 - Eğitim süreci state grafikleri")
        print("5 - Eğitim süreci ödül ve faz grafikleri")
        print("6 - Çift eğitim süreci animasyonu")
        print("7 - Çift eğitim süreci state grafikleri")
        print("8 - Çift eğitim süreci ödül ve faz grafikleri")
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
                visualizer.create_cart_pendulum_animation(visualizer.compress(states, compression_factor=STATE_COMPRESSION))

            elif choice == '2':
                file_path = os.path.join(base_path, "states_zero.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.plot_states(visualizer.compress(states, compression_factor=STATE_COMPRESSION), "Free Simulation")

            elif choice == '3':
                file_path = os.path.join(base_path, "states.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(file_path)
                visualizer.create_cart_pendulum_animation(visualizer.compress(states, compression_factor=STATE_COMPRESSION))

            elif choice == '4':
                reward_states_file = os.path.join(base_path, "reward_states.npy")
                if not os.path.exists(reward_states_file):
                    print(f"{reward_states_file} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                states = np.load(reward_states_file)
                visualizer.plot_states(visualizer.compress(states, compression_factor=STATE_COMPRESSION), "Training")

            elif choice == '5':
                rewards_file = os.path.join(base_path, "rewards.npy")
                reward_states_file = os.path.join(base_path, "reward_states.npy")
                if not os.path.exists(file_path):
                    print(f"{file_path} dosyası bulunamadı, dosyanın uygun konumda bulunduğundan emin olunuz.")
                    continue
                rewards = np.load(rewards_file)
                states = np.load(reward_states_file)
                visualizer.plot_phase_and_rewards(states, rewards)

            elif choice == '6':
                states_L = np.load(os.path.join(base_path, "states_L.npy"))
                states_R = np.load(os.path.join(base_path, "states_R.npy"))
                visualizer.create_dual_cart_pendulum_animation(visualizer.compress(states_L, compression_factor=STATE_COMPRESSION), visualizer.compress(states_R, compression_factor=STATE_COMPRESSION))

            elif choice == '7':
                states_L = np.load(os.path.join(base_path, "reward_states_L.npy"))
                states_R = np.load(os.path.join(base_path, "reward_states_R.npy"))
                visualizer.plot_states(visualizer.compress(states_L, compression_factor=STATE_COMPRESSION), "Double Training, Left Cart")
                visualizer.plot_states(visualizer.compress(states_R, compression_factor=STATE_COMPRESSION), "Double Training, Right Cart")

            elif choice == '8':
                states_L = np.load(os.path.join(base_path, "reward_states_L.npy"))
                states_R = np.load(os.path.join(base_path, "reward_states_R.npy"))
                rewards_L = np.load(os.path.join(base_path, "rewards_L.npy"))
                rewards_R = np.load(os.path.join(base_path, "rewards_R.npy"))

                # Sol araba grafikleri
                visualizer.plot_phase_and_rewards(visualizer.compress(states_L, compression_factor=STATE_COMPRESSION), visualizer.compress(rewards_L, compression_factor=REWARD_COMPRESSION))
                # Sağ araba grafikleri
                visualizer.plot_phase_and_rewards(visualizer.compress(states_R, compression_factor=STATE_COMPRESSION), visualizer.compress(rewards_R, compression_factor=REWARD_COMPRESSION))

            else:
                print("Geçersiz seçim! Lütfen 0-6 arasında bir sayı giriniz.")

        except Exception as e:
            print(f"İşlem sırasında bir hata oluştu: {e}")
            continue

if __name__ == "__main__":
    main()