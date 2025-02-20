import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths to Pickle Files
FEATURE_PATH = r"C:\glt works\Prem Ananth\code\traindb\features.pkl"
TARGET_PATH = r"C:\glt works\Prem Ananth\code\traindb\target.pkl"

# Load Feature and Target Data
def load_data(feature_path, target_path):
    with open(feature_path, "rb") as f:
        features = pickle.load(f)
    with open(target_path, "rb") as f:
        targets = pickle.load(f)
    
    return np.array(features), np.array(targets)

# Load Data
X, Y = load_data(FEATURE_PATH, TARGET_PATH)

# Normalize Features
X = X / np.max(X)  # Scale to [0,1]

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build PCNN Model (Fully Connected Neural Network)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),  # Input Layer
    Dropout(0.3),
    Dense(128, activation='relu'),  # Hidden Layer 1
    Dropout(0.2),
    Dense(64, activation='relu'),  # Hidden Layer 2
    Dense(1, activation='sigmoid')  # Output Layer (Binary Classification)
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32)

# Save Model
model.save("pulse_coupled_nn.h5")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('PCNN Training Performance')
plt.show()