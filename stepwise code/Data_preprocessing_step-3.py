import numpy as np
import os
from sklearn.model_selection import train_test_split

img_folder = '  ' # where all the npy files are saved

# Define the actions and their corresponding labels
actions = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","A", "B", "C", "D", "E", "F", "G", "H",
            "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] # List of actions

label_map = {action: idx for idx, action in enumerate(actions)}

# Initialize lists to hold keypoints and labels
X, y = [], []

# Set the number of landmarks for one hand (MediaPipe has 21 keypoints per hand)
# Modify this value depending on how many keypoints are in your data
keypoints_per_hand = 21

# Load keypoints data and create labels
for action in actions:
    action_folder = os.path.join(img_folder, action)
    for file_name in os.listdir(action_folder):
        if file_name.endswith('.npy'):
            # Load the keypoints
            keypoints = np.load(os.path.join(action_folder, file_name))

            # If the keypoints include both hands, slice only the first hand's keypoints
            if keypoints.shape[0] >= 2 * keypoints_per_hand:
                keypoints = keypoints[:keypoints_per_hand]  # Keep only the first hand's keypoints

            X.append(keypoints)
            y.append(label_map[action])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the keypoints
X = X / np.max(X)  # Normalize to the range [0, 1]

# Here we assume that you want to keep it as sequences of 1 step
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape to (samples, time steps, features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Preprocessing Complete.")
print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# Save the data as .npy files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
