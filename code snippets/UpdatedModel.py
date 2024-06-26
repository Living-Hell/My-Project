import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Preprocessing function
def preprocess_data(data, scaler=None, template_columns=None):
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    if template_columns is None:
        template_columns = data.columns

    data = pd.DataFrame(data_scaled, columns=template_columns)

    return data, scaler, template_columns

# Load the data
data = pd.read_csv('key_press_data.csv')

# # Method-1
# ascii_values={'Key.enter': 0, "'1'": 1, "'2'": 2, "'3'": 3, "'4'": 4, "'5'": 5, "'6'": 6, "'7'": 7, "'8'": 8, "'9'": 9, "'0'": 10, "'-'": 11, "'='": 12, 'Key.backspace': 13, 'Key.tab': 14, "'q'": 15, "'w'": 16, "'e'": 17, "'r'": 18, "'t'": 19, "'y'": 20, "'u'": 21, "'i'": 22, "'o'": 23, "'p'": 24, "'['": 25, "']'": 26, "'\\\\'": 27, 'Key.caps_lock': 28, "'a'": 29, "'s'": 30, "'d'": 31, "'f'": 32, "'g'": 33, "'h'": 34, "'j'": 35, "'k'": 36, "'l'": 37, "';'": 38, '"\'"': 39, 'Key.shift': 40, "'z'": 41, "'x'": 42, "'c'": 43, "'v'": 44, "'b'": 45, "'n'": 46, "'m'": 47, "','": 48, "'.'": 49, "'/'": 50, 'Key.shift_r': 51, 'Key.ctrl_l': 52, '<255>': 53, 'Key.cmd': 54, 'Key.alt_l': 55, 'Key.space': 56, 'Key.alt_gr': 57, 'Key.ctrl_r': 58, 'Key.left': 59, 'Key.up': 60, 'Key.down': 61, 'Key.right': 62, 'Key.esc': 63, "'!'": 64, "'@'": 65, "'#'": 66, "'$'": 67, "'%'": 68, "'^'": 69, "'&'": 70, "'*'": 71, "'('": 72, "')'": 73, "'_'": 74, "'+'": 75, "'{'": 76, "'}'": 77, "'|'": 78, "':'": 79, '\'"\'': 80, "'<'": 81, "'>'": 82, "'?'": 83, '<188>': 84, '<190>': 85, '<191>': 86, "'\\x0c'": 87, '<186>': 88, '<222>': 89, "'\\x10'": 90, "'\\x1b'": 91, "'\\x1d'": 92, "'\\x1c'": 93, "'A'": 94, "'Q'": 95, "'W'": 96, "'E'": 97, "'R'": 98, "'T'": 99, "'Y'": 100, "'U'": 101, "'I'": 102, "'O'": 103, "'P'": 104, "'L'": 105, "'K'": 106, "'J'": 107, "'H'": 108, "'G'": 109, "'F'": 110, "'D'": 111, "'S'": 112, "'Z'": 113, "'X'": 114, "'C'": 115, "'V'": 116, "'B'": 117, "'N'": 118, "'M'": 119}
# data['Key'] = data['Key'].map(ascii_values)

# Method-2
ascii_values = {}
def keyToInt(data):
  for key in data['Key']:
    if key not in ascii_values:
      ascii_values[key] = len(ascii_values)

keyToInt(data)
data['Key'] = data['Key'].map(ascii_values)


# Now data['Key'] contains integer values corresponding to each key
print(data.head())


preprocessed_data, scaler, template_columns = preprocess_data(data)
preprocessed_data.head()

# Split the data into training and test sets
X = preprocessed_data.drop(columns=['User ID'])
y = preprocessed_data['User ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure there are no infinite values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Ensure the data types are float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Build the enhanced autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 32  # Size of the encoding layer

autoencoder = models.Sequential([
    layers.InputLayer(input_shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Evaluate the model
reconstructions = autoencoder.predict(X_test)
reconstruction_errors = np.mean(np.square(reconstructions - X_test), axis=1)

# Set a threshold for anomaly detection
threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

def is_anomaly(new_data, autoencoder, threshold, scaler, template_columns):
    # Preprocess the new data with the same template columns and scaler
    new_data, _, _ = preprocess_data(new_data, scaler, template_columns)

    # Ensure new_data columns match template_columns
    for col in template_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    # Ensure the order of columns matches template_columns
    new_data = new_data[template_columns]

    # Convert to numpy array and ensure the data type is float32
    new_data = new_data.to_numpy().astype(np.float32)

    # Reshape the new data to match the expected input shape
    new_data = new_data[:, :autoencoder.input_shape[1]]  # Ensure correct number of features
    new_data = new_data.reshape(-1, autoencoder.input_shape[1])  # Reshape to match the model input shape

    # Predict using the autoencoder
    reconstructions = autoencoder.predict(new_data)
    reconstruction_errors = np.mean(np.square(reconstructions - new_data), axis=1)

    # Check if the reconstruction error exceeds the threshold
    anomalies = reconstruction_errors > threshold

    # Count the number of anomalies detected
    num_anomalies = np.sum(anomalies)

    return anomalies, num_anomalies


# Example usage
new_data = pd.read_csv('new_key_press_data.csv')

# #Method-1
# new_data['Key'] = new_data['Key'].map(ascii_values)

#Method-2
keyToInt(new_data)
new_data['Key'] = new_data['Key'].map(ascii_values)

anomalies, num_anomalies = is_anomaly(new_data, autoencoder, threshold, scaler, template_columns)

total_data_points = new_data.shape[0]
anomaly_percentage = (num_anomalies / total_data_points) * 100

print(f"Total data points: {total_data_points}")
print(f"Anomalies detected: {num_anomalies}")
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
if anomaly_percentage > 50:
    print("Anomaly detected, your system might have been taken over!")
else:
    print("No anomaly detected, you are safe.")
