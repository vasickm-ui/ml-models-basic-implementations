import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import matplotlib.pyplot as plt
import time

def generate_sequence():
    a = np.random.randint(1,101)
    b = np.random.randint(1,101)
    seq = [a,b]
    for _ in range(4):
        seq.append(seq[-1] + seq[-2])
        
    return seq

def create_dataset(n_samples):
    X=[]
    y=[]
    
    for _ in range(n_samples):
        seq = generate_sequence()
        X.append(seq[:5])
        y.append(seq[5])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    
    X = X.reshape((X.shape[0], 5, 1))
    return X, y

X_train, y_train = create_dataset(1000)
X_val, y_val = create_dataset(1000)

model = Sequential([
    SimpleRNN(16, activation='tanh', input_shape=(4,1)),
    Dense(1)    
])

model1 = Sequential([
    LSTM(16, activation='tanh', input_shape=(4,1)),
    LSTM(16, activation='tanh'),
    Dense(1)    
])

model.compile(optimizer='adam', loss='mse')
model1.compile(optimizer='adam', loss='mse')

class GradientsPerEpoch(tf.keras.callbacks.Callback):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.gradients_history = []
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            y_pred = self.model(self.x_data, training=False)
            loss = self.loss_fn(self.y_data, y_pred)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.gradients_history.append([g.numpy().copy() for g in grads])

        grad_norm = tf.linalg.norm(grads[0]).numpy()
        print(f"Epoch {epoch+1}: L2 norm of first parameter gradient = {grad_norm:.6f}")

grad_logger = GradientsPerEpoch(X_train, y_train)
grad_logger1 = GradientsPerEpoch(X_train, y_train)
history = model.fit(X_train, y_train, epochs=32, batch_size=32, verbose=1, callbacks=[grad_logger])
history1 = model1.fit(X_train, y_train, epochs=32, batch_size=32, verbose=1, callbacks=[grad_logger1])



loss = model.evaluate(X_val, y_val, verbose=0)
loss1 = model1.evaluate(X_val, y_val, verbose=0)
print("Loss: ", loss)
print("Loss 1: ", loss1)



num_epochs = len(grad_logger.gradients_history)
num_epochs1 = len(grad_logger1.gradients_history)
epochs = np.arange(1, num_epochs + 1)
epochs1 = np.arange(1, num_epochs1 + 1)
grad_norms = [np.linalg.norm(grads[0]) for grads in grad_logger.gradients_history]
grad_norms1 = [np.linalg.norm(grads[0]) for grads in grad_logger1.gradients_history]
plt.figure(figsize=(8,4))
plt.plot(epochs, grad_norms, marker='o', label="Vanilla RNN")
plt.plot(epochs1, grad_norms1, color='red', marker='x', label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("L2 norm of first parameter gradient")
plt.title("Comparison between Vanilla RNN and LSTM")
plt.grid(True)
plt.show()


