# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_points = 5000

# Coefficients
a1 = a2 = 5
a3 = a4 = a5 = a6 = -0.5

# Generate random samples from Uniform(0, 0.5)
Ut = np.random.uniform(0, 0.5, size=num_points)

# Generate the time series based on the specified model
Xt = np.zeros_like(Ut)
for t in range(6, num_points):
    Xt[t] = Ut[t] + a1 * Ut[t-1] + a2 * Ut[t-2] + a3 * Ut[t-3] + a4 * Ut[t-4] + a5 * Ut[t-5] + a6 * Ut[t-6]

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(Xt, label='Xt')
plt.title('Average Model')
plt.xlabel('Time')
plt.ylabel('Xt')
plt.legend()
plt.show()

# -



# # Sample of original series

# +
start_index = 100
end_index = 300


x_values = np.arange(start_index, end_index)
plt.figure(figsize=(10, 6))
plt.plot(x_values, Xt[start_index:end_index], label='Original Series')
# -











# # Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)

# +
import statsmodels.api as sm

# Plot autocorrelation and partial autocorrelation functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sm.graphics.tsa.plot_acf(Xt, lags=20, ax=ax1, title='Autocorrelation Function (ACF)')
sm.graphics.tsa.plot_pacf(Xt, lags=20, ax=ax2, title='Partial Autocorrelation Function (PACF)')
plt.show()


# -





# # Model building

# +
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Function to create input sequences and labels for training the RNN
def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)

# Define the number of time steps for input sequences
time_steps = 6


x, y = create_sequences(Xt, time_steps)


x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp,y_temp,test_size=1/8, random_state=42)


model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stopping])

loss = model.evaluate(x_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('RNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()


predictions = model.predict(x)

plt.figure(figsize=(10, 6))
plt.plot(Xt, label='Original Series')
plt.plot(np.arange(time_steps, len(Xt)), predictions, label='RNN Predictions')
plt.title('RNN Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Xt')
plt.legend(loc="upper right")
plt.show()

# -



# # Sample of original and predicted series

# +

start_index = 100
end_index = 300


x_values = np.arange(start_index, end_index)
plt.figure(figsize=(10, 6))
plt.plot(x_values, Xt[start_index:end_index], label='Original Series')
plt.plot(x_values, predictions[start_index:end_index], label='RNN Predictions')
plt.title('RNN Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Xt')
plt.legend(loc="upper right")
plt.show()

# -




























