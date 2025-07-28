import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Assuming X_train, y_train, X_test, y_test are already prepared and loaded
# Load the data from .npy files
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Confirm the shapes of the loaded data
print(f'Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'Loaded X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


# Define the model
model = Sequential()

# LSTM Layer
model.add(LSTM(64, return_sequences=True, input_shape=(1, 21)))  # Adjust the input shape based on your data
model.add(Dropout(0.5))  # Dropout for regularization

# Another LSTM Layer
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))

# Fully Connected Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Assuming 'model' is your trained LSTM model
model.save('sign_language_model.h5')
model.save('sign_language_model.keras')
