from datetime import datetime, timedelta
from queue import Queue

import cv2 as cv
import mediapipe as mp
import numpy as np

from Handlers.DrawingHandler import DrawingHandler
from Helpers.CvFpsCalc import CvFpsCalc
from Helpers.HandGestures import *


class VisionHandler:
    def __init__(self, device=0, cap_width=960, cap_height=540, use_static_image_mode=True,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5, communication_queue: Queue = Queue()):
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
        self.communication_queue = communication_queue

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
        self.go_last_detected = None

        # gesture variables
        self.prev_gesture = ""
        self.curr_gesture = ""

    def detect_twirl(self, landmarks, handedness):
        if detectBasic(landmarks, handedness):
            self.curr_time_twirl = datetime.now()
            self.count_twirl = 0

        if self.curr_time_twirl is not None and detectTwirlEnd(landmarks,
                                                               handedness) and self.curr_time_twirl + timedelta(
                seconds=2) > datetime.now():
            self.count_twirl += 1

        if self.count_twirl > 15:
            self.curr_gesture = "twirl"
            self.curr_time_twirl = None
            self.count_twirl = 0

    def detect_stop_go(self, landmarks, handedness):
        if detectClosed(landmarks, handedness):
            self.count_go += 1
        else:
            self.count_go = 0

        if self.count_go > 10 and (self.go_last_detected is None or self.go_last_detected + timedelta(seconds = 5) < datetime.now()):
            self.go = not self.go
            if self.go: self.curr_gesture = "stop"
            else: self.curr_gesture = "go"
            self.go_last_detected = datetime.now()
            self.count_go = 0

    def detect_swipe(self, landmarks, handedness):
        if self.curr_time_swipe is not None and datetime.now() > self.curr_time_swipe + timedelta(seconds=3):
            self.curr_time_swipe = None
            self.tracker_x = []
            self.tracker_y = []
            self.tracker_z = []
            self.distance = []
        else:
            if detectSideways(landmarks, handedness):
                if len(self.tracker_x) == 0:
                    self.curr_time_swipe = datetime.now()
                self.tracker_x.append(landmarks.landmark[8].x)
                self.tracker_y.append(landmarks.landmark[8].y)
                self.tracker_z.append(landmarks.landmark[8].z)

                # Compute the Euclidean distance between two landmarks
                distance = euclideanDistance(landmarks.landmark[12], landmarks.landmark[0])
                self.distance.append(distance)

                vari = variance(self.tracker_y)

                # a, b = np.polyfit(x, y, 1)
                x = np.array(self.tracker_x)
                y = np.array(self.tracker_y)
                A = np.vstack([x, np.ones(len(x))]).T
                a, b = np.linalg.lstsq(A, y, rcond=None)[0]

                mean_distance = np.mean(self.distance)

                if abs(np.max(self.tracker_x) - np.min(self.tracker_x)) >= 1.5 * mean_distance and vari < .002 and abs(a) < .15:
                    if np.sum(self.tracker_x[0:int(len(self.tracker_x) / 2)]) < np.sum(self.tracker_x[int(len(self.tracker_x) / 2):]):
                        self.curr_gesture = "swipe_left"
                    else:
                        self.curr_gesture = "swipe_right"
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

            if results.right_hand_landmarks is not None:
                landmarks = results.right_hand_landmarks
                image_rows, image_cols, _ = image.shape

                # detect gestures
                self.detect_twirl(landmarks, "R")
                self.detect_stop_go(landmarks, "R")
                self.detect_swipe(landmarks, "R")

                if self.curr_gesture is not None and self.curr_gesture != "":
                    self.communication_queue.put(("/gesture", self.curr_gesture))
                    # print(self.curr_gesture)
                    self.curr_gesture = None

                cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


            if results.left_hand_landmarks is not None:
                landmarks = results.left_hand_landmarks
                image_rows, image_cols, _ = image.shape
                # detect gestures
                self.detect_twirl(landmarks, "L")
                self.detect_stop_go(landmarks, "L")
                self.detect_swipe(landmarks, "L")

                if self.curr_gesture is not None and self.curr_gesture != "":
                    self.communication_queue.put(("/gesture", self.curr_gesture))
                    self.curr_gesture = None

                cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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
                if self.curr_gesture == 'wave':
                    self.communication_queue.put(("/live", (head_x, head_y, head_z, shoulder_x, shoulder_y, shoulder_z)))

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