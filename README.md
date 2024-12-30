# Inverted Pendulum Control Project with AI Learning Paradigms (Türkçe içerik için aşağı kaydırın) 

This repository contains a comprehensive control framework for inverted pendulum systems, employing both reinforcement learning and supervised learning approaches. The project aims to explore advanced control strategies while providing modular implementations and visualization tools.  

## Table of Contents  

- [Project Overview](#project-overview)  
- [Repository Structure](#repository-structure)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)
- [License](#license)  

## Project Overview  

The project explores two distinct approaches for controlling inverted pendulum systems:  
1. **Reinforcement Learning**:  
   - Single and competitive multi-agent scenarios  
   - Deep Q-Learning (DQN) implementation with adaptive exploration and advanced neural networks  

2. **Supervised Learning**:  
   - Linear Quadratic Regulator (LQR) for data generation  
   - Neural network implementation using both custom NumPy code and TensorFlow  

The primary goals include:  
- Stabilization of single and competitive pendulums  
- Modular and reusable implementations for research and development  
- Real-time visualization and data analysis  

## Repository Structure  

```
.
├── reinforcement-learning/      # RL-based control implementation
│   ├── pendulum_nonlinear_model.py    # Physics engine
│   ├── pendulum_training.py           # Single pendulum RL
│   ├── pendulum_training_fight.py     # Multi-agent competition
|   ├── pendulum_zero_simulation.py    # Zero force applied simulation 
│   └── pendulum_visualizer.py         # Visualization tools
│
└── supervised-learning/         # Supervised learning-based control
    ├── pendulum_lqr_control.py      # LQR data generation
    ├── pendulum_supervised_math.py  # Custom NN implementation
    ├── pendulum_supervised.py       # TensorFlow NN implementation
    └── pendulum_data.csv            # Generated training data
```

## Features  

### Reinforcement Learning  
- **Single Pendulum Control**: Stabilizes a single pendulum using DQN with advanced features such as experience replay and target network synchronization.  
- **Two-Pendulum Competition**: Simulates competitive interactions with balanced and attack forces.  

### Supervised Learning  
- **LQR Control**: Generates high-quality data for supervised training using state space models.  
- **Neural Network Training**: Implements both a custom NumPy-based neural network and a TensorFlow/Keras model for control.  

### Visualization Tools  
- Real-time animations and performance metrics  
- Force prediction and state space analysis  

## Installation  

Install the required dependencies:  

```bash
pip install numpy tensorflow scipy matplotlib control scikit-learn pandas
```

## Usage  

### Reinforcement Learning  

```bash
# Train single pendulum
python reinforcement-learning/pendulum_training.py

# Simulate two-pendulum competition
python reinforcement-learning/pendulum_training_fight.py
```

### Supervised Learning  

```bash
# Generate training data with LQR
python supervised-learning/pendulum_lqr_control.py

# Train neural network (NumPy implementation)
python supervised-learning/pendulum_supervised_math.py

# Train neural network (TensorFlow implementation)
python supervised-learning/pendulum_supervised.py
```

## Contributing

Contributions welcome! Please feel free to:
- Submit issues
- Fork the repository
- Create pull requests
- Suggest improvements

## License  

This project is licensed under the MIT License - see the LICENSE file for details.  

---

# Yapay Zeka Öğrenme Yaklaşımları ile Ters Sarkaç Kontrol Projesi

Bu depo, hem pekiştirmeli öğrenme hem de denetimli öğrenme yaklaşımlarını kullanarak ters sarkaç sistemleri için kapsamlı bir kontrol çerçevesi sunar. Proje, gelişmiş kontrol stratejilerini araştırmayı hedeflerken modüler uygulamalar ve görselleştirme araçları sağlar.

## İçindekiler  

- [Proje Genel Bakış](#proje-genel-bakış)  
- [Depo Yapısı](#depo-yapısı)  
- [Özellikler](#özellikler)  
- [Kurulum](#kurulum)  
- [Kullanım](#kullanım)  
- [Katkıda Bulunma](#katkıda-bulunma)  
- [Lisans](#lisans)  

## Proje Genel Bakış  

Proje, ters sarkaç sistemlerinin kontrolü için iki farklı yaklaşımı araştırır:  
1. **Pekiştirmeli Öğrenme**:  
   - Tekli ve rekabetçi çoklu ajan senaryoları  
   - Derin Q-Öğrenme (DQN) ile adaptif keşif ve gelişmiş sinir ağları  

2. **Denetimli Öğrenme**:  
   - Veri üretimi için Lineer Kuadratik Regülatör (LQR)  
   - Hem özel NumPy kodu hem de TensorFlow kullanılarak sinir ağı uygulaması  

Ana hedefler:  
- Tekli ve rekabetçi sarkaçların dengelemesi  
- Araştırma ve geliştirme için modüler ve yeniden kullanılabilir uygulamalar  
- Gerçek zamanlı görselleştirme ve veri analizi  

## Depo Yapısı  

```
.
├── reinforcement-learning/      # Pekiştirmeli öğrenme tabanlı kontrol uygulaması
│   ├── pendulum_nonlinear_model.py    # Fizik motoru
│   ├── pendulum_training.py           # Tek sarkaç pekiştirmeli öğrenme
│   ├── pendulum_training_fight.py     # Çoklu ajan rekabeti
|   ├── pendulum_zero_simulation.py    # Sıfır kuvvet uygulamalı simülasyon
│   └── pendulum_visualizer.py         # Görselleştirme araçları
│
└── supervised-learning/         # Denetimli öğrenme tabanlı kontrol
    ├── pendulum_lqr_control.py      # LQR veri üretimi
    ├── pendulum_supervised_math.py  # Özel sinir ağı uygulaması
    ├── pendulum_supervised.py       # TensorFlow sinir ağı uygulaması
    └── pendulum_data.csv            # Üretilen eğitim verisi
```

## Özellikler  

### Pekiştirmeli Öğrenme  
- **Tekli Sarkaç Kontrolü**: DQN ile tecrübe tekrar oynatma ve hedef ağ senkronizasyonu gibi gelişmiş özellikler kullanılarak tek bir sarkaç dengelenir.  
- **İkili Sarkaç Rekabeti**: Dengelenmiş ve saldırı kuvvetleriyle rekabetçi etkileşimler simüle edilir.  

### Denetimli Öğrenme  
- **LQR Kontrolü**: Durum uzayı modelleri kullanılarak denetimli eğitim için yüksek kaliteli veriler üretilir.  
- **Sinir Ağı Eğitimi**: Hem özel NumPy tabanlı hem de TensorFlow/Keras modeliyle kontrol uygulanır.  

### Görselleştirme Araçları  
- Gerçek zamanlı animasyonlar ve performans ölçümleri  
- Kuvvet tahmini ve durum uzayı analizi  

## Kurulum  

Gerekli bağımlılıkları yükleyin:  

```bash
pip install numpy tensorflow scipy matplotlib control scikit-learn pandas
```

## Kullanım  

### Pekiştirmeli Öğrenme  

```bash
# Tek sarkaç eğitimi
python reinforcement-learning/pendulum_training.py

# İkili sarkaç rekabeti simülasyonu
python reinforcement-learning/pendulum_training_fight.py
```

### Denetimli Öğrenme  

```bash
# LQR ile eğitim verisi üretimi
python supervised-learning/pendulum_lqr_control.py

# Sinir ağı eğitimi (NumPy uygulaması)
python supervised-learning/pendulum_supervised_math.py

# Sinir ağı eğitimi (TensorFlow uygulaması)
python supervised-learning/pendulum_supervised.py
```

## Katkıda Bulunma  

Katkılar memnuniyetle karşılanır! Lütfen şunları yapmaktan çekinmeyin:
- Sorun bildirme
- Depoyu çatallama
- Çekme istekleri oluşturma
- İyileştirme önerme  

## Lisans  

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.  