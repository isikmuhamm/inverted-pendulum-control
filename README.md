# Ters Sarkaç Kontrolü için Yapay Zeka Metodları Kullanımı (English Below)

Bu proje, inverted pendulum sisteminin kontrolü için reinforcement learning (RL) ve supervised learning (denetimli öğrenme) tekniklerinin uygulanmasını içermektedir. Amaç, farklı kontrol stratejilerini keşfetmek ve bu stratejilerin pendulumun dengede tutulmasındaki etkinliğini değerlendirmektir.

## Proje Yapısı

### Reinforcement Learning

- **`pendulum_nonlinear_model.py`**  
  Bu dosya, inverted pendulum'un diferansiyel denklemlerle oluşturulmuş bir modelini içermektedir. Pendulumun davranışını görselleştirmek için animasyonlu ve grafiksel çıktılar sağlamaktadır.

- **`pendulum_training.py`**  
  Bu betik, inverted pendulum'u kontrol etmek için Deep Q-Network (DQN) algoritmasını uygular. Sistem, reinforcement learning aracılığıyla öğrenme ve adaptasyon yeteneğine sahip olmaya çalışır.

### Supervised Learning

- **`pendulum_lqr_control.py`**  
  Bu dosya, Linear Quadratic Regulator (LQR) yöntemini kullanarak inverted pendulum'u kontrol etmektedir. Elde edilen sonuçlar bir CSV dosyasına kaydedilmektedir.

- **`pendulum_supervised_math.py`**  
  Bu betik, temel matematiksel fonksiyonları kullanarak model eğitmeye çalışmaktadır. Herhangi bir kütüphane kullanılmadan CSV verileri ile işlem yapar ve grafiksel çıktılar sağlar.

- **`pendulum_supervised.py`**  
  Bu dosya, güncel kütüphaneleri ve teknikleri kullanarak CSV verileri ile model eğitir ve sonuçları grafiksel olarak sunar.

## Kurulum ve Kullanım

### Gereksinimler

Projeyi çalıştırmak için Python 3.x sürümüne ve aşağıdaki kütüphanelere ihtiyacınız var:

- NumPy
- Matplotlib
- TensorFlow
- Keras

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu terminalde çalıştırın:

```bash
pip install numpy matplotlib tensorflow keras
```

### Script'lerin Çalıştırılması

1. **Reinforcement Learning için:**
   - Pendulum modelini çalıştırmak için:
     ```bash
     python pendulum_nonlinear_model.py
     ```
   - DQN ajanını eğitmek için:
     ```bash
     python pendulum_training.py
     ```

2. **Supervised Learning için:**
   - LQR ile kontrol sağlamak için:
     ```bash
     python pendulum_lqr_control.py
     ```
   - Temel matematik fonksiyonlarıyla model eğitmek için:
     ```bash
     python pendulum_supervised_math.py
     ```
   - Gelişmiş kütüphanelerle model eğitmek için:
     ```bash
     python pendulum_supervised.py
     ```

## Sonuçlar

Her yöntemin çıktıları ilgili CSV dosyalarında veya grafiksel gösterimlerde bulunmaktadır. Sonuçları inceleyerek farklı kontrol stratejilerinin etkinliğini karşılaştırabilirsiniz.

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, aşağıdaki adımları takip edebilirsiniz:

1. Repository'yi fork edin.
2. Yeni bir özellik eklemek veya hata düzeltmek için bir branch oluşturun:
   ```bash
   git checkout -b feature/ÖzellikAdı
   ```
3. Değişikliklerinizi yapın ve commit edin:
   ```bash
   git commit -m "Özellik eklendi"
   ```
4. Branch'ınızı GitHub'a push edin:
   ```bash
   git push origin feature/ÖzellikAdı
   ```
5. Pull request oluşturun.

## İletişim

Herhangi bir sorunuz veya geri bildiriminiz varsa, lütfen benimle iletişime geçin:

- **Email:** [isikmuhamm@gmail.com](mailto:isikmuhamm@gmail.com)
- **GitHub Profil:** [isikmuhamm](https://github.com/isikmuhamm)

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

## Teşekkürler

Reinforcement learning ve kontrol sistemleri alanındaki katkıda bulunanlara teşekkür ederiz.

---
  
  
# Use of Artificial Intelligence Methods for Inverted Pendulum Control

This project involves the application of reinforcement learning (RL) and supervised learning techniques for the control of the inverted pendulum system. The aim is to explore different control strategies and assess their effectiveness in maintaining the pendulum's balance.

## Project Structure

### Reinforcement Learning

- **`pendulum_nonlinear_model.py`**  
  This file contains a model of the inverted pendulum created through differential equations. It provides animated and graphical outputs to visualize the behavior of the pendulum.

- **`pendulum_training.py`**  
  This script implements the Deep Q-Network (DQN) algorithm to control the inverted pendulum. The system attempts to learn and adapt through reinforcement learning.

### Supervised Learning

- **`pendulum_lqr_control.py`**  
  This file controls the inverted pendulum using the Linear Quadratic Regulator (LQR) method. The results obtained are saved in a CSV file.

- **`pendulum_supervised_math.py`**  
  This script attempts to train a model using basic mathematical functions. It processes CSV data without using any libraries and provides graphical outputs.

- **`pendulum_supervised.py`**  
  This file trains a model using CSV data with current libraries and techniques, presenting the results graphically.

## Installation and Usage

### Requirements

You need Python 3.x and the following libraries to run the project:

- NumPy
- Matplotlib
- TensorFlow
- Keras

To install the required libraries, run the following command in the terminal:

```bash
pip install numpy matplotlib tensorflow keras
```

### Running the Scripts

1. **For Reinforcement Learning:**
   - To run the pendulum model:
     ```bash
     python pendulum_nonlinear_model.py
     ```
   - To train the DQN agent:
     ```bash
     python pendulum_training.py
     ```

2. **For Supervised Learning:**
   - To control using LQR:
     ```bash
     python pendulum_lqr_control.py
     ```
   - To train a model with basic mathematical functions:
     ```bash
     python pendulum_supervised_math.py
     ```
   - To train a model using advanced libraries:
     ```bash
     python pendulum_supervised.py
     ```

## Results

The outputs of each method can be found in the corresponding CSV files or graphical representations. You can compare the effectiveness of different control strategies by examining the results.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a branch for adding a new feature or fixing a bug:
   ```bash
   git checkout -b feature/FeatureName
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Feature added"
   ```
4. Push your branch to GitHub:
   ```bash
   git push origin feature/FeatureName
   ```
5. Create a pull request.

## Contact

If you have any questions or feedback, please feel free to contact me:

- **Email:** [isikmuhamm@gmail.com](mailto:isikmuhamm@gmail.com)
- **GitHub Profile:** [isikmuhamm](https://github.com/isikmuhamm)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to all contributors in the fields of reinforcement learning and control systems.