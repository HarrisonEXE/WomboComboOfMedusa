import cv2
import mediapipe as mp
import time
from pythonosc import udp_client
from gestures import *
import numpy as np
from datetime import datetime, timedelta

# UDP Client
client = udp_client.SimpleUDPClient("192.168.2.2", 12346)
# client = udp_client.SimpleUDPClient("0.0.0.0", 12346)

# Build Keypoint's using MP Holistic
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Holistic

def mediapipe_detection(image, model):
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


mp_kwargs = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# For webcam input:
cap = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

# Initialize gestures
count_wave = 0
curr_time_wave = datetime.now()
curr_time_wave_f = curr_time_wave + timedelta(seconds = 5)
waving = False

prev_gesture = ""
gesture = ""

with mp_holistic.Holistic(**mp_kwargs) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image, results = mediapipe_detection(image, holistic)
        draw_styled_landmarks(image, results)

        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks
            image_rows, image_cols, _ = image.shape
            movements = detect_hand_gesture(landmarks, "R")

            if movements.get("front") and movements.get("upright") and not movements.get("close"):
                    count_wave += 1

            if datetime.now() < curr_time_wave_f and count_wave >= 20:
                if not waving:
                    gesture = "wave_hello"
                    print("wave detected: HI")
                else:
                    gesture = "wave_bye"
                    print("wave detected: BYE")
                waving = not waving
                count_wave = 0
                curr_time_wave = datetime.now()
                curr_time_wave_f = curr_time_wave + timedelta(seconds = 5)
            elif datetime.now() >= curr_time_wave_f:
                curr_time_wave = datetime.now()
                curr_time_wave_f = curr_time_wave + timedelta(seconds = 5)
                count_wave = 0

            if gesture is not None:
                if gesture != prev_gesture:
                    client.send_message("/gesture", gesture)
                cv2.putText(image, str(gesture), (1700, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                prev_gesture = gesture

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark
            head_x = landmarks[mp_holistic.PoseLandmark.NOSE.value].x
            head_y = landmarks[mp_holistic.PoseLandmark.NOSE.value].y
            head_z = landmarks[mp_holistic.PoseLandmark.NOSE.value].z
            shoulder_x = (landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x) / 2
            shoulder_y = (landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y) / 2
            shoulder_z = (landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].z + landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z) / 2
            if prev_gesture == 'wave_hello':
                client.send_message("/head", [head_x, head_y, head_z, shoulder_x, shoulder_y, shoulder_z])
                print("Head: ", head_x, head_y)

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()