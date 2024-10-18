# Pekiştirmeli Öğrenme ile Ters Sarkaç Dengeleme Problemi (English Below)

Bu proje, bir ters sarkaç sistemini dengelemek için Derin Q-Learning (DQN) algoritmasını kullanmaktadır. DQN, takviye öğrenme (reinforcement learning) alanında kullanılan bir yöntemdir ve bir sinir ağı kullanarak eylem değerlerini (Q değerleri) tahmin eder. Projenin amacı, sarkacı dik tutmaya çalışırken en uygun kuvvetleri öğrenmektir.

## İçindekiler

- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Kod Yapısı](#kod-yapısı)
- [Algoritma Açıklaması](#algoritma-açıklaması)
- [Sonuçlar](#sonuçlar)

## Kurulum

Proje için gerekli olan kütüphaneleri yüklemek için aşağıdaki komutları kullanabilirsiniz:

```bash
pip install numpy matplotlib scipy tensorflow
```

## Kullanım

Projenin çalıştırılması için Python ortamında aşağıdaki komutu kullanın:

```bash
python pendulum_training.py
```
Projenin modelini animasyonlu bir şekilde izlemek için Python ortamında aşağıdaki komutu kullanın:

```bash
python pendulum_nonlinear_model.py
```

### Çıktılar

Program çalıştırıldığında, ters sarkaç sisteminin pozisyonu, hızı, açısı ve açısal hızı gibi durum bilgileri bir grafik ile gösterilecektir.

## Kod Yapısı

- `pendulum_dynamics(state, t, M, m, b, l, I, g, F)`: Ters sarkaç sisteminin dinamiklerini tanımlayan fonksiyon.
- `pendulum_step(state, F, time_step)`: Bir adımda sistemin yeni durumunu hesaplar.
- `DQNAgent`: DQN algoritmasını uygulayan ajan sınıfı.
  - `__init__`: Ajanın başlangıç parametrelerini ayarlar.
  - `_build_model`: Sinir ağı modelini oluşturur.
  - `update_target_model`: Hedef modelin ağırlıklarını günceller.
  - `act`: Mevcut duruma göre bir aksiyon seçer.
  - `remember`: Deneyimleri hafızaya kaydeder.
  - `replay`: Ajanın deneyimlerine göre modelini günceller.

## Algoritma Açıklaması

1. **Parametrelerin Tanımlanması**: Fiziksel özellikler (`M`, `m`, `b`, `l`, `I`, `g`) tanımlanır.
2. **Diferansiyel Denklem Çözümü**: `pendulum_dynamics` fonksiyonu, mevcut durum ve kuvvete göre hız ve hızlanmaları hesaplar.
3. **Ajan Oluşturma**: `DQNAgent` sınıfı, durum ve aksiyon uzayını alır ve Q değerlerini öğrenmek için bir sinir ağı oluşturur.
4. **Eğitim Süreci**: Her bölümde ajan, mevcut durumdan bir aksiyon seçer, yeni durumu hesaplar, ödül alır ve bu bilgiyi hafızaya kaydeder. Minibatch kullanarak ağı eğitir.
5. **Epsilon-Greedy Politikasının Uygulanması**: Ajan, keşif (exploration) ve sömürü (exploitation) arasında denge kurar. `epsilon` değeri zamanla azalır.
6. **Sonuçların Görselleştirilmesi**: Eğitim süreci sonunda sarkaç sisteminin durumu grafiklerle gösterilir.

## Sonuçlar

Proje sonunda, ters sarkaç sistemi başarılı bir şekilde dengede tutulmaya çalışılmakta ve eğitim sürecinin verimliliği grafikte gösterilmektedir. 

Grafikte, sarkacın pozisyonu ve açı gibi durumları gözlemlenebilir. 



----------------------------------------------------------------------------------------



# Inverted Pendulum Balancing with Reinforcement Control

This project uses the Deep Q-Learning (DQN) algorithm to balance an inverted pendulum system. DQN is a method used in the field of reinforcement learning and estimates action values (Q-values) using a neural network. The goal of the project is to learn the optimal forces while trying to keep the pendulum upright.

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Algorithm Explanation](#algorithm-explanation)
- [Results](#results)

## Installation

You can use the following commands to install the necessary libraries for the project:

```bash
pip install numpy matplotlib scipy tensorflow
```

## Usage

To run the project in a Python environment, use the following command:

```bash
python pendulum_training.py
```

To view the model of the project with animation, use the following command:

```bash
python pendulum_nonlinear_model.py
```

### Outputs

When the program is run, state information such as the position, velocity, angle, and angular velocity of the inverted pendulum system will be displayed in a graph.

## Code Structure

- `pendulum_dynamics(state, t, M, m, b, l, I, g, F)`: Function that defines the dynamics of the inverted pendulum system.
- `pendulum_step(state, F, time_step)`: Calculates the new state of the system in one step.
- `DQNAgent`: Class implementing the DQN algorithm.
  - `__init__`: Sets the initial parameters of the agent.
  - `_build_model`: Creates the neural network model.
  - `update_target_model`: Updates the weights of the target model.
  - `act`: Selects an action based on the current state.
  - `remember`: Stores experiences in memory.
  - `replay`: Updates the model based on the agent’s experiences.

## Algorithm Explanation

1. **Defining Parameters**: The physical properties (`M`, `m`, `b`, `l`, `I`, `g`) are defined.
2. **Differential Equation Solution**: The `pendulum_dynamics` function calculates the velocities and accelerations based on the current state and force.
3. **Agent Creation**: The `DQNAgent` class takes the state and action space and creates a neural network to learn Q-values.
4. **Training Process**: In each episode, the agent selects an action based on the current state, calculates the new state, receives a reward, and stores this information in memory. The agent trains the network using a minibatch.
5. **Applying the Epsilon-Greedy Policy**: The agent balances exploration and exploitation. The `epsilon` value decreases over time.
6. **Visualization of Results**: At the end of the training process, the state of the pendulum system is shown with graphs.

## Results

At the end of the project, the inverted pendulum system is successfully kept in balance, and the efficiency of the training process is shown in the graph.

In the graph, the states of the pendulum such as position and angle can be observed.