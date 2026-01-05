import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

start = time.time()

x = np.linspace(-20, 20, 700)
y = x**3

x = x.reshape(-1,1)
y = y.reshape(-1,1)

print(x)
print(x.shape)
print(tf.keras)

model = Sequential([
    Dense(12, activation='relu',input_shape=(1,)),
    Dense(12, activation='relu',input_shape=(1,)),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)



history = model.fit(
    x,y,
    epochs=300,
    verbose=0
)

x_val = np.linspace(-20,20,100).reshape(-1,1)
y_val = x_val ** 3

predictions = model.predict(x_val)


x_flat = x.flatten()
y_flat = y.flatten()
x_val_flat = x_val.flatten()
predictions_flat = predictions.flatten()

plt.figure(figsize=(6,4))
plt.scatter(x_flat, y_flat, color='blue', label='Training data')
plt.scatter(x_val_flat, predictions_flat, color='red', label='Predictions')
end = time.time()
print(f'Vreme izvrsavanja je: {end-start:.4f} s')
plt.show()





