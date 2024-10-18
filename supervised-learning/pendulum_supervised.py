import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Veri yükleme ve ön işleme
data = pd.read_csv('supervised-learning/pendulum_data.csv')
X = data[['x', 'x_dot', 'theta', 'theta_dot', 'r']].values
y = data['F'].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Model oluşturma ve eğitme
model = Sequential([
    Dense(20, activation='relu', input_shape=(5,)),
    Dense(20, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Model performansını değerlendirme
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Kuvvet (F)')
plt.ylabel('Tahmin Edilen Kuvvet (F)')
plt.title('Gerçek vs Tahmin Edilen Kuvvet')
plt.show()

# Eğitim sürecini görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Ortalama Kare Hatası')
plt.title('Model Eğitim Süreci')
plt.legend()
plt.show()

print("Model eğitimi ve değerlendirmesi tamamlandı. Grafikler gösteriliyor.")