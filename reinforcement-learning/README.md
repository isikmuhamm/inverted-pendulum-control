# Inverted Pendulum Control Project with Reinforcement Learning (Türkçe içerik için aşağı kaydırın)

This project implements advanced control strategies for an inverted pendulum system, featuring both single-pendulum reinforcement learning and competitive two-pendulum scenarios. Using Deep Q-Learning (DQN), the system learns to maintain pendulums in their upright positions while managing cart movements and competitive interactions.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [System Model](#system-model)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Data Storage](#data-storage)
- [Results and Visualization](#results-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project uses Deep Q-Learning to control inverted pendulum systems. Key objectives include:
- Single pendulum stabilization
- Competitive two-pendulum scenarios
- Real-time visualization and analysis
- Advanced reinforcement learning implementation

## Project Structure

```
.
└── reinforcement-learning/       # Model and simulation data storage
    ├── pendulum_nonlinear_model.py     # Core physics engine
    ├── pendulum_zero_simulation.py      # Zero-force simulation
    ├── pendulum_training.py             # Single pendulum RL
    ├── pendulum_training_fight.py       # Competitive simulation
    └── pendulum_visualizer.py           # Visualization tools
```

## System Model

Based on the CTMS Michigan model, the system includes:

- Cart mass (M): 0.5 kg
- Pendulum mass (m): 0.2 kg
- Friction coefficient (b): 0.1 N/m/sec
- Pendulum length (l): 0.3 m
- Moment of inertia (I): 0.006 kg.m²
- Time step: 35ms

## Features

### 1. Zero-Force Simulation
- Natural system dynamics simulation
- Initial condition response analysis
- State tracking and visualization

### 2. Single Pendulum Control
- DQN implementation with:
  - Double DQN architecture
  - Experience replay buffer
  - Adaptive exploration rate
  - Customizable neural network

### 3. Two-Pendulum Competition
- Competitive agent interaction
- Balanced and attack force applications
- Optional Poisson impact distribution
- State and reward sharing

### 4. Visualization Tools
- Real-time animations
- State space analysis
- Training metrics
- Phase portraits

## Installation

```bash
# Clone the repository
git clone https://github.com/isikmuhamm/inverted-pendulum-control

# Install dependencies
pip install numpy tensorflow scipy matplotlib
```

## Usage

### Core Components

```bash
# Zero-force simulation
python reinforcement-learning/pendulum_zero_simulation.py

# Single pendulum training
python reinforcement-learning/pendulum_training.py

# Two-pendulum competition
python reinforcement-learning/pendulum_training_fight.py

# Visualization interface
python reinforcement-learning/pendulum_visualizer.py
```

### Visualization Options

1. Zero-force simulation animation
2. System state graphs
3. Training process visualization
4. State space analysis
5. Reward tracking
6. Live trained agent simulation

## Algorithm Details

### DQN Implementation
1. **State Space**: Includes cart position, velocity, pendulum angle, and angular velocity
2. **Action Space**: Discretized force values for balance and attack
3. **Training Process**:
   - Experience collection
   - Minibatch sampling
   - Q-value updates
   - Target network synchronization

### Key Components
- `DQNAgent`: Main learning agent class
- `pendulum_dynamics`: System physics implementation
- `pendulum_step`: State progression calculator
- `replay`: Experience-based learning method

## Configuration

Adjustable parameters:
```python
CONTINUE_TRAINING = False  # Continue from saved model
POISSON_IMPACTS = False   # Enable random impacts
max_len = 50000          # Replay buffer size
poisson_lambda = 10      # Impact frequency
```

## Model Architecture

```
Input Layer (4/8 neurons)
    ↓
Dense Layer (256 neurons, ReLU)
    ↓
Dense Layer (256 neurons, ReLU)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Output Layer (Action space size, Linear)
```

## Data Storage

Generated files:
- Model weights: `pendulum_model.keras`
- Agent state: `agent_state.npy`
- Training history: `states.npy`, `rewards.npy`

## Results and Visualization

The visualization system provides:
- Real-time system state monitoring
- Training progress tracking
- Performance metrics analysis
- Phase space visualization

## Contributing

Contributions welcome! Please feel free to:
- Submit issues
- Fork the repository
- Create pull requests
- Suggest improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

-----------------------------------

# Pekiştirmeli Öğrenme ile Ters Sarkaç Kontrol Projesi  

Bu proje, tekli sarkaç pekiştirmeli öğrenme ve rekabetçi iki sarkaç senaryolarını içeren gelişmiş bir kontrol stratejisini uygular. Deep Q-Learning (DQN) kullanarak sistem, sarkaçları dikey konumda tutmayı, araba hareketlerini yönetmeyi ve rekabetçi etkileşimleri öğrenir.  

## İçindekiler  

- [Proje Genel Bakış](#proje-genel-bakış)  
- [Proje Yapısı](#proje-yapısı)  
- [Sistem Modeli](#sistem-modeli)  
- [Özellikler](#özellikler)  
- [Kurulum](#kurulum)  
- [Kullanım](#kullanım)  
- [Algoritma Detayları](#algoritma-detayları)  
- [Yapılandırma](#yapılandırma)  
- [Model Mimarisi](#model-mimarisi)  
- [Veri Depolama](#veri-depolama)  
- [Sonuçlar ve Görselleştirme](#sonuçlar-ve-görselleştirme)  
- [Katkıda Bulunma](#katkıda-bulunma)  
- [Lisans](#lisans)  

## Proje Genel Bakış  

Bu proje, ters sarkaç sistemlerini kontrol etmek için Deep Q-Learning kullanır. Ana hedefler:  
- Tekli sarkaç dengeleme  
- Rekabetçi iki sarkaç senaryoları  
- Gerçek zamanlı görselleştirme ve analiz  
- Gelişmiş pekiştirmeli öğrenme uygulamaları  

## Proje Yapısı  

```  
.  
└── reinforcement-learning/       # Model ve simülasyon veri deposu  
    ├── pendulum_nonlinear_model.py     # Temel fizik motoru  
    ├── pendulum_zero_simulation.py      # Sıfır kuvvet simülasyonu  
    ├── pendulum_training.py             # Tekli sarkaç RL  
    ├── pendulum_training_fight.py       # Rekabetçi simülasyon  
    └── pendulum_visualizer.py           # Görselleştirme araçları  
```  

## Sistem Modeli  

CTMS Michigan modeline dayalı sistem şunları içerir:  
- Araba kütlesi (M): 0.5 kg  
- Sarkaç kütlesi (m): 0.2 kg  
- Sürtünme katsayısı (b): 0.1 N/m/sn  
- Sarkaç uzunluğu (l): 0.3 m  
- Atalet momenti (I): 0.006 kg.m²  
- Zaman adımı: 35ms  

## Özellikler  

### 1. Sıfır Kuvvet Simülasyonu  
- Doğal sistem dinamiği simülasyonu  
- Başlangıç durumu yanıt analizi  
- Durum takibi ve görselleştirme  

### 2. Tekli Sarkaç Kontrolü  
- DQN uygulaması ile:  
  - Çift DQN mimarisi  
  - Deney tekrarı tamponu  
  - Uyarlanabilir keşif oranı  
  - Özelleştirilebilir sinir ağı  

### 3. İki Sarkaç Rekabeti  
- Rekabetçi ajan etkileşimi  
- Dengeli ve saldırı kuvveti uygulamaları  
- Opsiyonel Poisson darbe dağılımı  
- Durum ve ödül paylaşımı  

### 4. Görselleştirme Araçları  
- Gerçek zamanlı animasyonlar  
- Durum uzayı analizi  
- Eğitim metrikleri  
- Faz portreleri  

## Kurulum  

```bash  
# Depoyu klonlayın  
git clone https://github.com/isikmuhamm/inverted-pendulum-control  

# Bağımlılıkları yükleyin  
pip install numpy tensorflow scipy matplotlib  
```  

## Kullanım  

### Temel Bileşenler  

```bash  
# Sıfır kuvvet simülasyonu  
python reinforcement-learning/pendulum_zero_simulation.py  

# Tekli sarkaç eğitimi  
python reinforcement-learning/pendulum_training.py  

# İki sarkaç rekabeti  
python reinforcement-learning/pendulum_training_fight.py  

# Görselleştirme arayüzü  
python reinforcement-learning/pendulum_visualizer.py  
```  

### Görselleştirme Seçenekleri  

1. Sıfır kuvvet simülasyonu animasyonu  
2. Sistem durumu grafikleri  
3. Eğitim süreci görselleştirmesi  
4. Durum uzayı analizi  
5. Ödül takibi  
6. Eğitilmiş ajan canlı simülasyonu  

## Algoritma Detayları  

### DQN Uygulaması  
1. **Durum Uzayı**: Araba pozisyonu, hızı, sarkaç açısı ve açısal hızı içerir  
2. **Aksiyon Uzayı**: Denge ve saldırı için ayrık kuvvet değerleri  
3. **Eğitim Süreci**:  
   - Deney toplama  
   - Minibatch örnekleme  
   - Q-değer güncellemeleri  
   - Hedef ağ senkronizasyonu  

### Ana Bileşenler  
- `DQNAgent`: Ana öğrenme ajan sınıfı  
- `pendulum_dynamics`: Sistem fiziği uygulaması  
- `pendulum_step`: Durum ilerleme hesaplayıcısı  
- `replay`: Deney tabanlı öğrenme yöntemi  

## Yapılandırma  

Ayarlanabilir parametreler:  
```python  
CONTINUE_TRAINING = False  # Kayıtlı modelden devam et  
POISSON_IMPACTS = False   # Rastgele darbeleri etkinleştir  
max_len = 50000          # Tekrar tampon boyutu  
poisson_lambda = 10      # Darbe sıklığı  
```  

## Model Mimarisi  

```  
Giriş Katmanı (4/8 nöron)  
    ↓  
Yoğun Katman (256 nöron, ReLU)  
    ↓  
Yoğun Katman (256 nöron, ReLU)  
    ↓  
Yoğun Katman (128 nöron, ReLU)  
    ↓  
Çıkış Katmanı (Aksiyon uzayı boyutu, Doğrusal)  
```  

## Veri Depolama  

Oluşturulan dosyalar:  
- Model ağırlıkları: `pendulum_model.keras`  
- Ajan durumu: `agent_state.npy`  
- Eğitim geçmişi: `states.npy`, `rewards.npy`  

## Sonuçlar ve Görselleştirme  

Görselleştirme sistemi şunları sağlar:  
- Gerçek zamanlı sistem durumu takibi  
- Eğitim ilerlemesi takibi  
- Performans metrikleri analizi  
- Faz uzayı görselleştirme  

## Katkıda Bulunma  

Katkılar memnuniyetle karşılanır! Lütfen:  
- Sorunlar gönderin  
- Depoyu çatallayın  
- Çekme istekleri oluşturun  
- İyileştirme önerileri sunun  

## Lisans  

Bu proje MIT Lisansı ile lisanslanmıştır - detaylar için LICENSE dosyasına bakın.