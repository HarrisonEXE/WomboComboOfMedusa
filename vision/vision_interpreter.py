"""
Author: Hardik Goel
Date: 2023-02-02
Model Guide: https://google.github.io/mediapipe/solutions/holistic.html
Description: This script is used to run the vision interpreter.  Uses MediaPipe Holistic model to detect pose, hands,
and face. The Holistic model is a single, multitask model that simultaneously detects: 543 landmarks (33 pose
landmarks, 468 face landmarks, and 21 hand landmarks per hand). We will use this model to detect body pose and hands
only.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from pythonosc import udp_client
from gestures import *
from enum import Enum
from datetime import datetime, timedelta

# UDP Client
client = udp_client.SimpleUDPClient("192.168.2.2", 12346)

# Build Keypoint's using MP Holistic
mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Holistic


class Gesture(Enum):
    WAVE = 1
    TWIRL = 2
    STOP = 3


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
    # Draw face connections
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    # Draw pose connections
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
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

# gesture counters/ timers
count_twirl = 0
curr_time_twirl = datetime.now()
count_wave = 0
curr_time_wave = datetime.now()
curr_time_wave_f = curr_time_wave + timedelta(seconds = 5)
waving = False
count_clap = 0
curr_time_clap = datetime.now()
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
        draw_landmarks(image, results)

        if results.pose_landmarks:
            movements = detect_pose(results.pose_landmarks)
            if movements.get("bow"):
                gesture = "bow"
                print(gesture)
            if gesture is not None:
                if gesture != prev_gesture:
                    client.send_message("/gesture", gesture)
                    print('sent')
                cv2.putText(image, str(gesture), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            prev_gesture = gesture

        if results.left_hand_landmarks and results.right_hand_landmarks:
            image_rows, image_cols, _ = image.shape
            movements = detect_hand_gesture(results, "Both")
            if movements.get("clap"):
                gesture = "clap"
                print(gesture)
            if gesture is not None:
                if gesture != prev_gesture:
                    client.send_message("/gesture", gesture)
                    print('sent')
                cv2.putText(image, str(gesture), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            prev_gesture = gesture

        elif results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks
            image_rows, image_cols, _ = image.shape
            movements = detect_hand_gesture(landmarks, "R")

            if movements.get("front") and movements.get("upright"):
                curr_time_twirl = datetime.now()
                count_twirl = 0
                # print("start")
                # print(count_twirl)
                if not movements.get("close"):
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

            if movements.get("back") and (curr_time_twirl + timedelta(seconds=1.0) > datetime.now()):
                count_twirl += 1
                # print("counting")
                # print(count_twirl)

            if count_twirl > 7:
                count_twirl = 0
                print("complete")
                gesture = "twirl"
                print(gesture)

            if gesture is not None:
                if gesture != prev_gesture:
                    client.send_message("/gesture", gesture)
                    print('sent')
                cv2.putText(image, str(gesture), (1700, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            prev_gesture = gesture

        elif results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks
            image_rows, image_cols, _ = image.shape
            movements = detect_hand_gesture(landmarks, "L")

            if movements.get("front") and movements.get("upright"):
                curr_time_twirl = datetime.now()
                count_twirl = 0
                # print("start")
                # print(count_twirl)
                if not movements.get("close"):
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

            if movements.get("back") and (curr_time_twirl + timedelta(seconds=1.0) > datetime.now()):
                count_twirl += 1
                # print("counting")
                # print(count_twirl)

            if count_twirl > 7:
                count_twirl = 0
                # print("complete")
                gesture = "twirl"
                print(gesture)

            if gesture is not None:
                if gesture != prev_gesture:
                    client.send_message("/gesture", gesture)
                cv2.putText(image, str(gesture), (10, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            prev_gesture = gesture

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
