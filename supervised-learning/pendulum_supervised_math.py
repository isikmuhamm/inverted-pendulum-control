import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid fonksiyonu ve türevi
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# İleri besleme
def forward_propagation(X, weights, biases):
    Z1 = np.dot(X, weights['W1']) + biases['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, weights['W2']) + biases['b2']
    A2 = Z2  # Çıkışta lineer aktivasyon
    return Z1, A1, Z2, A2

# Geri yayılım
def back_propagation(X, y, Z1, A1, A2, weights, biases, learning_rate):
    m = X.shape[0]
    
    # Hatalar
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2) / m
    
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    
    # Ağırlıklar ve biasları güncelleme
    weights['W2'] -= learning_rate * dW2
    biases['b2'] -= learning_rate * db2
    weights['W1'] -= learning_rate * dW1
    biases['b1'] -= learning_rate * db1

# Maliyet fonksiyonu (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Yapay sinir ağı eğitimi
def train_neural_network(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs):
    np.random.seed(42)
    
    # Ağırlıklar ve biasları rastgele başlatma
    weights = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'W2': np.random.randn(hidden_size, output_size) * 0.01
    }
    biases = {
        'b1': np.zeros((1, hidden_size)),
        'b2': np.zeros((1, output_size))
    }
    
    # Eğitim döngüsü
    loss_history = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X_train, weights, biases)
        loss = mean_squared_error(y_train, A2)
        loss_history.append(loss)
        
        back_propagation(X_train, y_train, Z1, A1, A2, weights, biases, learning_rate)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    return weights, biases, loss_history

# Tahmin yapma
def predict(X, weights, biases):
    _, _, _, A2 = forward_propagation(X, weights, biases)
    return A2

# Veri yükleme
data = pd.read_csv('supervised-learning/pendulum_data.csv')

# Özellikler ve hedef
X = data[['x', 'x_dot', 'theta', 'theta_dot', 'r']].values
y = data['F'].values.reshape(-1, 1)

# Veriyi eğitim ve test setlerine ayırma
train_size = int(0.8 * X.shape[0])
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Yapay sinir ağı parametreleri
input_size = X_train.shape[1]
hidden_size = 20  # Gizli katman boyutu
output_size = 1  # Tek bir çıktı (Kuvvet F)
learning_rate = 0.05
epochs = 2000

# Modeli eğitme
weights, biases, loss_history = train_neural_network(X_train, y_train, input_size, hidden_size, output_size, learning_rate, epochs)

# Test verisinde tahmin yapma
y_pred_test = predict(X_test, weights, biases)

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Kuvvet (F)')
plt.ylabel('Tahmin Edilen Kuvvet (F)')
plt.title('Gerçek vs Tahmin Edilen Kuvvet')
plt.show()

# Eğitim sürecini görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(loss_history, label='Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Ortalama Kare Hatası')
plt.title('Model Eğitim Süreci')
plt.legend()
plt.show()

print("Model eğitimi ve değerlendirmesi tamamlandı. Grafikler gösteriliyor.")
