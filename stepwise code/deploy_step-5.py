import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp  # Import MediaPipe for hand landmark extraction

# Load your trained model
model = tf.keras.models.load_model('sign_language_model.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract keypoints from the video frame
def extract_keypoints_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = []

    if results.multi_hand_landmarks:
        # We are only taking the keypoints of one hand (the first detected hand)
        for landmark in results.multi_hand_landmarks[0].landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
    
    # Return None if no keypoints were found or if less than 21 landmarks (63 values) are found
    return keypoints if len(keypoints) == 63 else None

# Initialize video capture
cap = cv2.VideoCapture(0)

actions = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","A", "B", "C", "D", "E", "F", "G", "H",
           "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] # List of actions

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Extract keypoints from the frame
    keypoints = extract_keypoints_from_frame(frame)

    if keypoints is not None:
        keypoints_array = np.array(keypoints).flatten()

        # Pad or truncate keypoints_array to match expected size (21) for one hand
        if len(keypoints_array) < 21:
            keypoints_array = np.pad(keypoints_array, (0, 21 - len(keypoints_array)), 'constant')
        else:
            keypoints_array = keypoints_array[:21]

        # Reshape the array to match input shape (1, 1, 21) for the model
        keypoints_array = keypoints_array.reshape(1, 1, 21)

        # Make prediction with your model
        prediction = model.predict(keypoints_array)
        predicted_label_index = np.argmax(prediction)
        predicted_label = actions[predicted_label_index]

        # Log the prediction probabilities for debugging
        print(f"Predicted probabilities: {prediction}")

        # Overlay predicted label on the video frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with the prediction
    cv2.imshow('Sign Language Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
