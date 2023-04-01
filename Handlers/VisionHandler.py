
from pythonosc import udp_client
from datetime import datetime, timedelta
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from Handlers.DrawingHandler import DrawingHandler
from Helpers.CvFpsCalc import CvFpsCalc
from Helpers.HandGestures import *

# # UDP Client
# global client
# PORT = 12346
# IP = "192.168.2.2"
# client = udp_client.SimpleUDPClient(IP, PORT)


# TODO: Add wave detection
# TODO: Check stop/go detection client message
# TODO: Standardize gesture messages to be sent to client

class VisionHandler:
    def __init__(self, device=0, cap_width=960, cap_height=540, use_static_image_mode=True,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.cap = cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        self.drawingHandler = DrawingHandler(use_brect=True)

        self.holistic = mp.solutions.holistic
        self.pose = self.holistic.Holistic(
            static_image_mode=use_static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

    def start(self):
        self.run()
        self.cap.release()
        cv.destroyAllWindows()

    def initialize_gesture_detection_state(self):
        # twirl variables
        self.count_twirl = 0
        self.curr_time_twirl = None

        # swipe variables
        self.curr_time_swipe = None
        self.tracker_x = []
        self.tracker_y = []
        self.tracker_z = []
        self.distance = []

        # stop/go variables
        self.count_go = 0
        self.go = False

        # gesture variables
        self.prev_gesture = ""
        self.curr_gesture = ""

    def detect_twirl(self, landmarks, results):
        if detectBasic(landmarks, results):
            self.curr_time_twirl = datetime.now()
            self.count_twirl = 0

        if self.curr_time_twirl is not None and detectTwirlEnd(landmarks, results) and self.curr_time_twirl + timedelta(
                seconds=2) > datetime.now():
            self.count_twirl += 1

        if self.count_twirl > 15:
            # client.send_message("/gesture", 3)
            self.curr_gesture = "twirl"
            print("twirl now")
            time.sleep(5)
            self.count_twirl = 0

    def detect_stop_go(self, landmarks, results):
        if detectClosed(landmarks, results):
            self.count_go += 1
        else:
            self.count_go = 0

        if self.count_go > 10:
            self.go = not self.go
            print("stop/go detected")
            time.sleep(5)
            self.count_go = 0

    def detect_swipe(self, landmarks, results):
        if self.curr_time_swipe is not None and datetime.now() > self.curr_time_swipe + timedelta(seconds=3):
            self.curr_time_swipe = None
            self.tracker_x = []
            self.tracker_y = []
            self.tracker_z = []

        else:
            if detectSideways(landmarks, results):
                if len(self.tracker_x) == 0:
                    self.curr_time_swipe = datetime.now()
                self.tracker_x.append(landmarks.landmark[8].x)
                self.tracker_y.append(landmarks.landmark[8].y)
                self.tracker_z.append(landmarks.landmark[8].z)
                self.distance.append(euclideanDistance(landmarks.landmark[12], landmarks.landmark[0]))

                vari = variance(self.tracker_y)

                x = np.array(self.tracker_x)
                y = np.array(self.tracker_y)

                a, b = np.polyfit(x, y, 1)

                mean_distance = sum(self.distance) / len(self.distance)

                if abs(max(self.tracker_x) - min(self.tracker_x)) >= 1.5 * mean_distance and vari < .002 and abs(
                        a) < .15:
                    if sum(self.tracker_x[0:int(len(self.tracker_x) / 2)]) < sum(
                            self.tracker_x[int(len(self.tracker_x) / 2):]):
                        print("swipe right")
                        self.curr_gesture = "swipe_right"
                        # client.send_message("/swipe", 0)
                    else:
                        print("swipe left")
                        self.curr_gesture = "swipe_left"
                        # client.send_message("/swipe", 1)
                    self.tracker_x = []
                    self.tracker_y = []
                    self.tracker_z = []
                    self.distance = []
                    vari, x, y, a, b, mean_distance, self.curr_time_swipe = None, None, None, None, None, None, None

    def run(self):
        self.initialize_gesture_detection_state()

        while True:
            if cv.waitKey(5) & 0xFF == 27:
                break

            fps = self.cvFpsCalc.get()

            ret, frame = self.cap.read()
            image, results = self.mediapipe_detection(frame, self.pose)

            self.drawingHandler.draw_styled_landmarks(image, results)

            if results.right_hand_landmarks:
                landmarks = results.right_hand_landmarks
                image_rows, image_cols, _ = image.shape

                # detect gestures
                self.detect_twirl(landmarks, "R")
                self.detect_stop_go(landmarks, "R")
                self.detect_swipe(landmarks, "R")

                if self.curr_gesture is not None:
                    if self.curr_gesture != "":
                        print(self.curr_gesture)
                    # if self.curr_gesture != self.prev_gesture:
                    #     client.send_message("/gesture", self.curr_gesture)
                    cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    self.prev_gesture = self.curr_gesture

            if results.left_hand_landmarks:
                landmarks = results.right_hand_landmarks
                image_rows, image_cols, _ = image.shape

                # detect gestures
                self.detect_twirl(landmarks, "L")
                self.detect_stop_go(landmarks, "L")
                self.detect_swipe(landmarks, "L")

                if self.curr_gesture is not None:
                    if self.curr_gesture != "":
                        print(self.curr_gesture)
                    # if self.curr_gesture != self.prev_gesture:
                    #     client.send_message("/gesture", self.curr_gesture)
                    cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    self.prev_gesture = self.curr_gesture

            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                head_x = landmarks[self.holistic.PoseLandmark.NOSE.value].x
                head_y = landmarks[self.holistic.PoseLandmark.NOSE.value].y
                head_z = landmarks[self.holistic.PoseLandmark.NOSE.value].z
                shoulder_x = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].x) / 2
                shoulder_y = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].y) / 2
                shoulder_z = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].z + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].z) / 2
                # TODO: This will never be true since we don't have wave_hello anymore!!!
                # if self.prev_gesture == 'wave_hello':
                #     client.send_message("/live", [head_x, head_y, head_z, shoulder_x, shoulder_y, shoulder_z])
                    # print("Head: ", head_x, head_y)
                    # print("Shoulder: ", shoulder_x, shoulder_y)
            cv.putText(image, str(int(fps)) + " FPS", (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Gesture Recognition', image)

    def mediapipe_detection(self, image, model):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = model.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results
