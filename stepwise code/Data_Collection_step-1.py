import os
import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize Mediapipe drawing and hands modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to perform Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False                   # Mark the image as not writeable to improve performance
    results = model.process(image)                  # Make detections
    image.flags.writeable = True                    # Mark the image as writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results

# Function to draw styled landmarks on the image
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    keypoints = []
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract keypoints for each detected hand
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            keypoints.append(rh)
    
    # If only one hand is detected, pad with zeros for the second hand
    if len(keypoints) == 1:
        keypoints.append(np.zeros(21 * 3))  # 21 landmarks * (x, y, z)
    
    # If no hands are detected, return zeros for both hands
    if len(keypoints) == 0:
        keypoints = [np.zeros(21 * 3), np.zeros(21 * 3)]  # Two hands

    # Append the number of detected hands as an additional feature (1 or 2)
    num_hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    keypoints = np.concatenate(keypoints)
    keypoints = np.append(keypoints, num_hands_detected)  # Append hand count as last feature

    return keypoints

# Initialize Mediapipe Hands with settings
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

"""["A", "B", "C", "D", "E", "F", "G", "H", "I",
 "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", 
 "T", "U", "V", "W", "X", "Y", "Z"]

 ["0", "1", "2", "3", "4", 
 "5", "6", "7", "8", "9"]
"""

# Set up folder for saving images and actions
img_folder = ' ' # path where to save files 

actions =  ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","A", "B", "C", "D", "E", "F", "G", "H",
            "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] # List of actions

# Initialize camera
camera = cv2.VideoCapture(0)

# Flags to control capture state
capturing = False
JPG_capturing = False
no_of_frame_action = 1000  # Number of frames to capture per action

# Main loop for actions
for action in actions:
    # Reset capturing flags and counters for each action
    capturing = False
    JPG_capturing = False
    frame_count = 0
    JPG_frame_count = 0

    # Create folder for the current action
    action_folder = os.path.join(img_folder, action)
    os.makedirs(action_folder, exist_ok=True)
    print(f"Folder created: {action}")
    print("Press 's' to start capturing frames...")

    while True:
        ret, frame = camera.read()  # Read from video feed
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB for Mediapipe
        image, results = mediapipe_detection(frame, hands)

        # Draw landmarks on the frame (optional)
        draw_styled_landmarks(image, results)

        # Extract keypoints (for one or two hands)
        keypoints = extract_keypoints(results)

        # Display the frame with landmarks
        cv2.imshow('MediaPipe Hands', image)

        key = cv2.waitKey(1) & 0xFF

        # Start capturing when 's' is pressed
        if key == ord("s") and not capturing:
            capturing = True
            JPG_capturing = True
            print("Started capturing frames...")
            time.sleep(2)  # Allow time to position hands

        # Capture and save keypoints (as .npy files)
        if capturing:
            if frame_count < no_of_frame_action:
                npy_path = os.path.join(action_folder, f'{frame_count}.npy')
                np.save(npy_path, keypoints)
                frame_count += 1

                if frame_count == no_of_frame_action:
                    print(f"Done capturing frames for action {action}...")
                    capturing = False

        # Capture and save jpg images (for visualization)
        if JPG_capturing:
            if JPG_frame_count < 11:
                jpg_path = os.path.join(action_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(jpg_path, frame)
                JPG_frame_count += 1
                if JPG_frame_count == 11:
                    print("Done capturing JPG frames")
                    JPG_capturing = False

        # Stop capturing if 'p' is pressed
        if key == ord("p"):
            print(f"Stopped capturing for action {action}")
            capturing = False
            JPG_capturing = False

        # Move to the next action after completing the frames or pressing 'p'
        if frame_count >= no_of_frame_action or key == ord("p"):
            print(f"Moving to the next action after completing")
            break

        # Exit the program if 'f' is pressed
        if key == ord("f"):
            print("Exiting...")
            camera.release()
            cv2.destroyAllWindows()
            exit()  # Ensure the program exits cleanly

# Release resources after exiting loops
camera.release()
cv2.destroyAllWindows()
