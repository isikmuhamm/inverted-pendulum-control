# Inverted Pendulum Control Project with Supervised Learning (Türkçe içerik için aşağı kaydırın)

This project implements a supervised learning approach to control an inverted pendulum system. The project utilizes both classical control theory (LQR) for data generation and neural networks for learning the control strategy.

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

The project uses supervised learning to control an inverted pendulum system. Key objectives include:
- Data generation using LQR control
- Neural network implementation from scratch using NumPy
- TensorFlow/Keras implementation for comparison
- Real-time visualization and analysis

## Project Structure

```
.
├── supervised-learning/          # Model and data storage
    ├── pendulum_lqr_control.py      # LQR controller for data generation
    ├── pendulum_supervised_math.py   # Neural network from scratch
    ├── pendulum_supervised.py        # TensorFlow implementation
    └── pendulum_data.csv            # Generated training data
```

## System Model

The inverted pendulum system parameters include:
- Cart mass (M): 0.5 kg
- Pendulum mass (m): 0.2 kg
- Friction coefficient (b): 0.1 N/m/sec
- Moment of inertia (I): 0.006 kg⋅m²
- Gravity (g): 9.8 m/s²
- Pendulum length (l): 0.3 m

## Features

### 1. LQR Control Data Generation
- State space model implementation
- LQR controller design
- Multiple reference signal simulation
- Comprehensive data collection

### 2. NumPy Neural Network Implementation
- Custom neural network from scratch
- Forward and backward propagation
- Gradient descent optimization
- MSE loss calculation

### 3. TensorFlow/Keras Implementation
- Sequential model architecture
- Optimized training process
- Data standardization
- Performance comparison

### 4. Visualization Tools
- Force prediction comparison
- Training loss monitoring
- Real vs. Predicted visualization
- Performance metrics

## Installation

```bash
# Install required packages
pip install numpy pandas tensorflow matplotlib control scikit-learn
```

## Usage

### Core Components

```bash
# Generate training data
python supervised-learning/pendulum_lqr_control.py

# Train neural network (NumPy implementation)
python supervised-learning/pendulum_supervised_math.py

# Train neural network (TensorFlow implementation)
python supervised-learning/pendulum_supervised.py
```

## Algorithm Details

### LQR Implementation
1. **State Space**: Cart position, velocity, pendulum angle, and angular velocity
2. **Control Design**: Optimal LQR gain calculation
3. **Data Generation**: Multiple reference signals and responses

### Neural Network Implementation
1. **NumPy Version**:
   - Custom gradient descent
   - Sigmoid activation for hidden layer
   - Linear activation for output
   - MSE loss function

2. **TensorFlow Version**:
   - Adam optimizer
   - ReLU activations
   - StandardScaler preprocessing
   - Validation split

## Configuration

System parameters can be adjusted in `pendulum_lqr_control.py`:
```python
M, m, b, I, g, l = 0.5, 0.2, 0.1, 0.006, 9.8, 0.3
```

Neural network parameters:
```python
hidden_size = 20       # Hidden layer neurons
learning_rate = 0.05   # Learning rate
epochs = 2000         # Training epochs
```

## Model Architecture

### NumPy Implementation:
```
Input Layer (5 neurons)
    ↓
Hidden Layer (20 neurons, Sigmoid)
    ↓
Output Layer (1 neuron, Linear)
```

### TensorFlow Implementation:
```
Input Layer (5 neurons)
    ↓
Hidden Layer 1 (20 neurons, ReLU)
    ↓
Hidden Layer 2 (20 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
```

## Data Storage

Generated files:
- Training data: `pendulum_data.csv`
- Contains: time, states, control inputs

## Results and Visualization

Both implementations provide:
- Real vs. Predicted force comparison plots
- Training loss history visualization
- Performance metrics display
- Model accuracy analysis

## Contributing

Contributions welcome! Please feel free to:
- Submit issues
- Fork the repository
- Create pull requests
- Suggest improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

-----------------------------------

# Öğreticili Öğrenme ile Ters Sarkaç Kontrol Projesi  

Bu proje, ters sarkaç sistemini kontrol etmek için bir gözetimli öğrenme yaklaşımı uygular. Proje, veri üretimi için klasik kontrol teorisini (LQR) ve kontrol stratejisini öğrenmek için sinir ağlarını kullanır.  

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

Bu proje, ters sarkaç sistemini kontrol etmek için gözetimli öğrenmeyi kullanır. Ana hedefler:  
- LQR kontrolü ile veri üretimi  
- NumPy kullanarak sıfırdan sinir ağı uygulaması  
- Karşılaştırma için TensorFlow/Keras uygulaması  
- Gerçek zamanlı görselleştirme ve analiz  

## Proje Yapısı  

```  
.  
├── supervised-learning/          # Model ve veri deposu  
    ├── pendulum_lqr_control.py      # Veri üretimi için LQR kontrolcüsü  
    ├── pendulum_supervised_math.py   # Sıfırdan sinir ağı uygulaması  
    ├── pendulum_supervised.py        # TensorFlow uygulaması  
    └── pendulum_data.csv            # Üretilen eğitim verisi  
```  

## Sistem Modeli  

Ters sarkaç sistemi parametreleri:  
- Araba kütlesi (M): 0.5 kg  
- Sarkaç kütlesi (m): 0.2 kg  
- Sürtünme katsayısı (b): 0.1 N/m/sn  
- Atalet momenti (I): 0.006 kg⋅m²  
- Yer çekimi (g): 9.8 m/s²  
- Sarkaç uzunluğu (l): 0.3 m  

## Özellikler  

### 1. LQR Kontrolü ile Veri Üretimi  
- Durum uzayı modeli uygulaması  
- LQR kontrol tasarımı  
- Birden fazla referans sinyal simülasyonu  
- Kapsamlı veri toplama  

### 2. NumPy Sinir Ağı Uygulaması  
- Sıfırdan özel sinir ağı  
- İleri ve geri yayılım  
- Gradyan iniş optimizasyonu  
- MSE kayıp hesaplaması  

### 3. TensorFlow/Keras Uygulaması  
- Sıralı model mimarisi  
- Optimize edilmiş eğitim süreci  
- Veri standardizasyonu  
- Performans karşılaştırması  

### 4. Görselleştirme Araçları  
- Kuvvet tahmin karşılaştırması  
- Eğitim kaybı takibi  
- Gerçek vs. Tahmin görselleştirmesi  
- Performans metrikleri  

## Kurulum  

```bash  
# Gerekli paketleri yükleyin  
pip install numpy pandas tensorflow matplotlib control scikit-learn  
```  

## Kullanım  

### Temel Bileşenler  

```bash  
# Eğitim verisi üretimi  
python supervised-learning/pendulum_lqr_control.py  

# Sinir ağını eğitme (NumPy uygulaması)  
python supervised-learning/pendulum_supervised_math.py  

# Sinir ağını eğitme (TensorFlow uygulaması)  
python supervised-learning/pendulum_supervised.py  
```  

## Algoritma Detayları  

### LQR Uygulaması  
1. **Durum Uzayı**: Araba pozisyonu, hızı, sarkaç açısı ve açısal hızı  
2. **Kontrol Tasarımı**: Optimum LQR kazancı hesaplama  
3. **Veri Üretimi**: Birden fazla referans sinyali ve yanıtları  

### Sinir Ağı Uygulaması  
1. **NumPy Versiyonu**:  
   - Özel gradyan inişi  
   - Gizli katman için sigmoid aktivasyonu  
   - Çıkış için doğrusal aktivasyon  
   - MSE kayıp fonksiyonu  

2. **TensorFlow Versiyonu**:  
   - Adam optimizasyonu  
   - ReLU aktivasyonları  
   - StandardScaler ön işleme  
   - Doğrulama bölmesi  

## Yapılandırma  

Sistem parametreleri `pendulum_lqr_control.py` dosyasında ayarlanabilir:  
```python  
M, m, b, I, g, l = 0.5, 0.2, 0.1, 0.006, 9.8, 0.3  
```  

Sinir ağı parametreleri:  
```python  
hidden_size = 20       # Gizli katman nöronları  
learning_rate = 0.05   # Öğrenme oranı  
epochs = 2000         # Eğitim döngüleri  
```  

## Model Mimarisi  

### NumPy Uygulaması:  
```  
Giriş Katmanı (5 nöron)  
    ↓  
Gizli Katman (20 nöron, Sigmoid)  
    ↓  
Çıkış Katmanı (1 nöron, Doğrusal)  
```  

### TensorFlow Uygulaması:  
```  
Giriş Katmanı (5 nöron)  
    ↓  
Gizli Katman 1 (20 nöron, ReLU)  
    ↓  
Gizli Katman 2 (20 nöron, ReLU)  
    ↓  
Çıkış Katmanı (1 nöron, Doğrusal)  
```  

## Veri Depolama  

Üretilen dosyalar:  
- Eğitim verisi: `pendulum_data.csv`  
- İçerik: zaman, durumlar, kontrol girdileri  

## Sonuçlar ve Görselleştirme  

Her iki uygulama da şunları sağlar:  
- Gerçek vs. Tahmin edilen kuvvet karşılaştırma grafikleri  
- Eğitim kaybı geçmişi görselleştirme  
- Performans metrikleri ekranı  
- Model doğruluk analizi  

## Katkıda Bulunma  

Katkılar memnuniyetle karşılanır! Lütfen:  
- Sorunlar gönderin  
- Depoyu çatallayın  
- Çekme istekleri oluşturun  
- İyileştirme önerileri sunun  

## Lisans  

Bu proje MIT Lisansı ile lisanslanmıştır - detaylar için LICENSE dosyasına bakın.  