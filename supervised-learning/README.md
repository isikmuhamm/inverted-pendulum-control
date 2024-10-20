### Öğreticili Öğrenme ile Ters Sarkaç Dengeleme Problemi (English Below)

Bu proje, bir LQR (Doğrusal-Kare Regülatör) kontrolcüsü tarafından dengelenen bir araba-sarkaç sistemine öğretici öğrenme tekniklerini uygular. Amaç, sarkacı dengelemek için gereken kontrol kuvvetini tahmin edebilen bir yapay sinir ağı eğitmektir.

## İçindekiler

- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Kod Yapısı](#kod-yapısı)
- [Algoritma Açıklaması](#algoritma-açıklaması)
- [Sonuçlar](#sonuçlar)

## Kurulum

Gerekli kütüphaneleri aşağıdaki komutla kurabilirsiniz:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras control
```

## Kullanım

Projeyi çalıştırıp sarkaç verilerini simüle etmek için önce aşağıdaki komutu çalıştırın:

```bash
python pendulum_lqr_control.py
```

Veri oluşturulduktan sonra sinir ağı modellerini eğitmek ve değerlendirmek için:

```bash
python pendulum_supervised.py
```

Alternatif olarak, TensorFlow kullanmadan basit bir sinir ağı eğitmek isterseniz:

```bash
python pendulum_supervised_math.py
```

### Çıktılar

Programı çalıştırdığınızda, durum ve kontrol kuvveti verilerini içeren bir CSV dosyası (`pendulum_data.csv`) oluşturulur. Eğitim süreci, tahmin doğruluğu ve eğitim kaybı üzerine grafiklerle sonuçları gösterir.

## Kod Yapısı

- **`pendulum_lqr_control.py`**: LQR kontrolcüsü tarafından kontrol edilen araba-sarkaç sistemini simüle eder ve veri oluşturur.
  - `A`, `B`, `C`, `D`: Sistem için durum uzayı matrisleri.
  - `ctrl.lqr(A, B, Q, R)`: LQR kazançlarını hesaplar.
  - `sys_cl`: Kapalı çevrim sistem simülasyonu.
  - Simülasyon sonuçları daha sonra öğretici öğrenmede kullanılmak üzere CSV dosyasına kaydedilir.

- **`pendulum_supervised_math.py`**: Temel Python kütüphaneleri kullanarak manuel geri yayılım ile basit bir ileri beslemeli yapay sinir ağı uygular.
  - `sigmoid`, `sigmoid_derivative`: Aktivasyon fonksiyonu ve türevi.
  - `train_neural_network`: Geri yayılım kullanarak modeli eğitir.
  - `predict`: Verilen durumlar için kontrol kuvvetini tahmin eder.
  - Tahmin edilen ve gerçek kuvvet sonuçlarını görselleştirir.

- **`pendulum_supervised.py`**: TensorFlow/Keras kullanarak bir yapay sinir ağı eğitir.
  - İki gizli katmanlı `Sequential` model.
  - `StandardScaler`: Giriş ve çıkış verilerini ölçeklendirme.
  - `train_test_split`: Verileri eğitim ve test setlerine ayırır.
  - Model performansını ve eğitim kaybını epoch'lar boyunca görselleştirir.

## Algoritma Açıklaması

1. **LQR Veri Oluşturma**:
   - Sistem dinamiklerini tanımlayan `A`, `B`, `C` ve `D` durum uzayı matrisleri kullanılır.
   - LQR kontrolcüsü, durumu ve kontrol girdilerini içeren bir maliyet fonksiyonunu minimize eden optimal bir kontrol yasası hesaplar.
   - Sistem 50 adım boyunca simüle edilir ve sonuçlar `pendulum_data.csv` dosyasına kaydedilir.

2. **Öğretici Öğrenme**:
   - Amaç, sistem durumlarına (`x`, `x_dot`, `theta`, `theta_dot`, `r`) dayanarak sarkacı dengelemek için gereken kontrol kuvvetini (`F`) tahmin etmektir.
   
   - **Manuel Yapay Sinir Ağı**:
     - `pendulum_supervised_math.py` dosyasında bir gizli katmanlı basit bir yapay sinir ağı NumPy ile uygulanır.
     - Geri yayılım algoritması ile model eğitilir ve Ortalama Kare Hatası (MSE) kaybı minimize edilir.

   - **Keras Modeli**:
     - `pendulum_supervised.py` dosyası, ReLU aktivasyon fonksiyonlarını kullanan iki gizli katmanlı bir modeli TensorFlow/Keras ile eğitir.
     - `adam` optimizasyon algoritması, MSE kaybını minimize eder.
     - Modelin tahminleri gerçek değerlerle karşılaştırılır ve hem eğitim hem doğrulama kayıpları takip edilir.

## Sonuçlar

Proje sonunda, yapay sinir ağları başarıyla sistem durumlarına dayalı olarak kontrol kuvvetini (`F`) tahmin etmeyi öğrenir. Hem manuel hem de Keras tabanlı sinir ağları değerlendirilir ve sonuçlar görselleştirilir:

- Gerçek ve tahmin edilen kuvvet değerlerini (`F`) gösteren bir dağılım grafiği.
- Keras tabanlı modelin öğrenme sürecini gösteren eğitim kaybı ve doğrulama kaybı grafikleri.

Öğretici öğrenme yaklaşımı, LQR tabanlı kontrolcüyü yaklaşık olarak taklit eder ve dinamik sistemlerde öğrenme tabanlı kontrolün potansiyelini gösterir.


----------------------------------------------


### Inverted Pendulum Balancing with Supervised Control

This project applies supervised learning techniques to control a cart-pendulum system using data generated by a Linear-Quadratic Regulator (LQR) controller. The goal is to train a neural network that can predict the control force required to balance the pendulum.

## Contents

- [Setup](#setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Algorithm Description](#algorithm-description)
- [Results](#results)

## Setup

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras control
```

## Usage

To run the project and simulate the pendulum data, first run the following command:

```bash
python pendulum_lqr_control.py
```

Once the data has been generated, train and evaluate the neural network models by running:

```bash
python pendulum_supervised.py
```

Alternatively, to train a simple neural network without TensorFlow, you can run:

```bash
python pendulum_supervised_math.py
```

### Outputs

Running the program will produce a CSV file (`pendulum_data.csv`) containing the state and control force data. The training process will generate visualizations of the prediction accuracy and the training loss over time.

## Code Structure

- **`pendulum_lqr_control.py`**: Simulates the cart-pendulum system controlled by an LQR controller and generates data.
  - `A`, `B`, `C`, `D`: State-space matrices of the system.
  - `ctrl.lqr(A, B, Q, R)`: Computes the LQR gains.
  - `sys_cl`: Closed-loop system simulation.
  - Simulation results are saved to a CSV file for later use in supervised learning.

- **`pendulum_supervised_math.py`**: Implements a custom feedforward neural network with manual backpropagation using basic Python libraries.
  - `sigmoid`, `sigmoid_derivative`: Activation function and its derivative.
  - `train_neural_network`: Trains the model using a custom backpropagation method.
  - `predict`: Predicts the control force for given states.
  - Results are visualized using matplotlib, showing predicted vs actual force.

- **`pendulum_supervised.py`**: Trains a neural network using TensorFlow/Keras.
  - `Sequential` model with two hidden layers.
  - `StandardScaler`: Scales the input and output data for improved training performance.
  - `train_test_split`: Splits the data into training and testing sets.
  - Visualizes the model's performance and the training loss over epochs.

## Algorithm Description

1. **LQR Data Generation**:
   - The state-space matrices `A`, `B`, `C`, and `D` describe the dynamics of the cart-pendulum system.
   - The Linear-Quadratic Regulator (LQR) computes an optimal control law that minimizes a quadratic cost function involving the state and control inputs.
   - The system is simulated for 50 steps with varying reference inputs (`r`), and the resulting data is stored in `pendulum_data.csv`.

2. **Supervised Learning**:
   - The goal is to use supervised learning to predict the control force (`F`) needed to balance the pendulum, based on the system states (`x`, `x_dot`, `theta`, `theta_dot`, `r`).
   
   - **Custom Neural Network**:
     - A basic neural network with one hidden layer is implemented manually in `pendulum_supervised_math.py` using NumPy.
     - The network is trained with a custom backpropagation algorithm, and the Mean Squared Error (MSE) loss is minimized.

   - **Keras Model**:
     - `pendulum_supervised.py` uses the TensorFlow/Keras framework to train a more sophisticated model with two hidden layers, using ReLU activation functions.
     - The `adam` optimizer is used to minimize the MSE loss.
     - The model's predictions are compared

 against the true values, and both training and validation losses are tracked.

## Results

At the end of the project, the neural networks successfully learn to predict the control force (`F`) based on the system's states. Both custom and Keras-based neural networks are evaluated, and the results are visualized:

- A scatter plot showing the actual vs predicted force values (`F`).
- Training loss and validation loss plots for the Keras-based model, displaying the model’s learning process.

The supervised learning approach provides a good approximation of the LQR-based controller, demonstrating the potential for learning-based control in dynamic systems.