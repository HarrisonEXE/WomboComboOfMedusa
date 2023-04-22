import time
from datetime import datetime, timedelta
from queue import Queue

import cv2 as cv
import mediapipe as mp

from Helpers.DataFilters import buffered_smooth, save_vision_data
from Handlers.DrawingHandler import DrawingHandler
from Helpers.CvFpsCalc import CvFpsCalc
from Helpers.HandGestures import *


class VisionHandler:
    def __init__(self, device=4, cap_width=960, cap_height=540, use_static_image_mode=True,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5, communication_queue: Queue = Queue()):
        self.cap = cv.VideoCapture(device)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

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
        self.count_twirl_left = 0
        self.curr_time_twirl_left = None
        self.count_twirl_right = 0
        self.curr_time_twirl_right = None

        #volume variables
        self.vol_start = False
        self.vol_origin = None
        self.vol_init = None

        # swipe variables
        self.curr_time_swipe_right = None
        self.tracker_x_right = []
        self.tracker_y_right = []
        self.tracker_z_right = []
        self.distance_right = []

        self.curr_time_swipe_left = None
        self.tracker_x_left = []
        self.tracker_y_left = []
        self.tracker_z_left = []
        self.distance_left = []

        # stop/go variables
        self.count_go_left = 0
        self.count_go_right = 0
        self.go = False
        self.go_last_detected = None

        # gesture variables
        self.prev_gesture = ""
        self.curr_gesture = ""
        self.if_tracking = False # set live tracking on/off

        # Noise filter variables
        self.head_x = []
        self.head_y = []
        self.head_z = []
        self.shoulder_x = []
        self.shoulder_y = []
        self.shoulder_z = []

        # Queue timeout variable
        self.last_queued = None

        self.last_queued = None

    def detect_twirl(self, landmarks, handedness):
        if handedness == "L":
            if detectBasic(landmarks, handedness):
                self.curr_time_twirl_left = datetime.now()
                self.count_twirl_left = 0

            if self.curr_time_twirl_left is not None and detectTwirlEnd(landmarks,
                                                                handedness) and self.curr_time_twirl_left + timedelta(
                    seconds=2) > datetime.now():
                self.count_twirl_left += 1

            if self.count_twirl_left > 15:
                self.curr_gesture = "twirl"
                self.curr_time_twirl_left = None
                self.count_twirl_left = 0
        if handedness == "R":
            if detectBasic(landmarks, handedness):
                self.curr_time_twirl_right = datetime.now()
                self.count_twirl_right = 0

            if self.curr_time_twirl_right is not None and detectTwirlEnd(landmarks,
                                                                handedness) and self.curr_time_twirl_right + timedelta(
                    seconds=2) > datetime.now():
                self.count_twirl_right += 1

            if self.count_twirl_right > 15:
                self.curr_gesture = "twirl"
                self.curr_time_twirl_right = None
                self.count_twirl_right = 0


    def detect_stop_go(self, landmarks, handedness):

        if (handedness == "R"):
            if detectClosed(landmarks, handedness):
                self.count_go_right += 1
            else:
                self.count_go_right = 0
        elif (handedness == "L"):
            if detectClosed(landmarks, handedness):
                self.count_go_left += 1
            else:
                self.count_go_left = 0

        if (self.count_go_left > 10 or self.count_go_right > 10) and (self.go_last_detected is None or self.go_last_detected + timedelta(seconds = 5) < datetime.now()):
            self.go = not self.go
            if self.go: self.curr_gesture = "stop"
            else: self.curr_gesture = "go"
            self.go_last_detected = datetime.now()
            self.count_go_left, self.count_go_right = 0, 0

    def detect_swipe(self, landmarks, handedness):
        if handedness == "R":
            if self.curr_time_swipe_right is not None and datetime.now() > self.curr_time_swipe_right + timedelta(seconds=3):
                self.curr_time_swipe_right = None
                self.tracker_x_right = []
                self.tracker_y_right = []
                self.tracker_z_right = []
                self.distance_right = []
            else:
                if detectSideways(landmarks, handedness):
                    if len(self.tracker_x_right) == 0:
                        self.curr_time_swipe_right = datetime.now()
                    self.tracker_x_right.append(landmarks.landmark[8].x)
                    self.tracker_y_right.append(landmarks.landmark[8].y)
                    self.tracker_z_right.append(landmarks.landmark[8].z)

                    # Compute the Euclidean distance between two landmarks
                    distance_right = euclideanDistance(landmarks.landmark[12], landmarks.landmark[0])
                    self.distance_right.append(distance_right)

                    vari_right = variance(self.tracker_y_right)

                    # a, b = np.polyfit(x, y, 1)
                    x_right = np.array(self.tracker_x_right)
                    y_right = np.array(self.tracker_y_right)
                    A_right = np.vstack([x_right, np.ones(len(x_right))]).T
                    a_right, b_right = np.linalg.lstsq(A_right, y_right, rcond=None)[0]

                    mean_distance_right = np.mean(self.distance_right)

                    if abs(np.max(self.tracker_x_right) - np.min(self.tracker_x_right)) >= 1.5 * mean_distance_right and vari_right < .002 and abs(a_right) < .15:
                        if np.sum(self.tracker_x_right[0:int(len(self.tracker_x_right) / 2)]) < np.sum(self.tracker_x_right[int(len(self.tracker_x_right) / 2):]):
                            self.curr_gesture = "swipe_left"
                        else:
                            self.curr_gesture = "swipe_right"
                        print("swiped")
                        self.tracker_x_right = []
                        self.tracker_y_right = []
                        self.tracker_z_right = []
                        self.distance_right = []
                        vari_right, x_right, y_right, a_right, b_right, mean_distance_right, self.curr_time_swipe = None, None, None, None, None, None, None
        if handedness == "L":
            if self.curr_time_swipe_left is not None and datetime.now() > self.curr_time_swipe_left + timedelta(seconds=3):
                self.curr_time_swipe_left = None
                self.tracker_x_left = []
                self.tracker_y_left = []
                self.tracker_z_left = []
                self.distance_left = []
            else:
                if detectSideways(landmarks, handedness):
                    if len(self.tracker_x_left) == 0:
                        self.curr_time_swipe_left = datetime.now()
                    self.tracker_x_left.append(landmarks.landmark[8].x)
                    self.tracker_y_left.append(landmarks.landmark[8].y)
                    self.tracker_z_left.append(landmarks.landmark[8].z)

                    # Compute the Euclidean distance between two landmarks
                    distance_left = euclideanDistance(landmarks.landmark[12], landmarks.landmark[0])
                    self.distance_left.append(distance_left)

                    vari_left = variance(self.tracker_y_left)

                    # a, b = np.polyfit(x, y, 1)
                    x_left = np.array(self.tracker_x_left)
                    y_left = np.array(self.tracker_y_left)
                    A_left = np.vstack([x_left, np.ones(len(x_left))]).T
                    a_left, b_left = np.linalg.lstsq(A_left, y_left, rcond=None)[0]

                    mean_distance_left = np.mean(self.distance_left)

                    if abs(np.max(self.tracker_x_left) - np.min(self.tracker_x_left)) >= 1.5 * mean_distance_left and vari_left < .002 and abs(a_left) < .15:
                        if np.sum(self.tracker_x_left[0:int(len(self.tracker_x_left) / 2)]) < np.sum(self.tracker_x_left[int(len(self.tracker_x_left) / 2):]):
                            self.curr_gesture = "swipe_left"
                        else:
                            self.curr_gesture = "swipe_right"
                        print("swiped")
                        self.tracker_x_left = []
                        self.tracker_y_left = []
                        self.tracker_z_left = []
                        self.distance_left = []
                        vari_left, x_left, y_left, a_left, b_left, mean_distance_left, self.curr_time_swipe = None, None, None, None, None, None, None

    def detect_xy_control(self, hand_landmarks, handedness):
        gestureCorrect = detectBasic(hand_landmarks, handedness)

        if gestureCorrect:
            if self.vol_origin is None:
                self.vol_origin = hand_landmarks.landmark[9]
            else:
                dist = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[12])
                if hand_landmarks.landmark[9].y < self.vol_origin.y - dist:
                    self.vol_origin = hand_landmarks.landmark[9]
                    self.curr_gesture = "up"
                if hand_landmarks.landmark[9].y > self.vol_origin.y + dist:
                    self.vol_origin = hand_landmarks.landmark[9]
                    self.curr_gesture = "down"
        else:
            self.vol_origin = None

    def run(self):
        self.initialize_gesture_detection_state()

        while True:
            key_pressed = cv.waitKey(10) & 0xFF
            if key_pressed == 27:
                break
            elif key_pressed == ord('d'):
                self.vol_start = not self.vol_start
                print("key pressed - tracking drums", self.vol_start)
            elif key_pressed == ord('t'):
                self.if_tracking = not self.if_tracking
                print("key pressed - live tracking", self.if_tracking)

            fps = self.cvFpsCalc.get()

            ret, frame = self.cap.read()
            image, results = self.mediapipe_detection(frame, self.pose)

            self.drawingHandler.draw_styled_landmarks(image, results)

            if results.right_hand_landmarks is not None:
                landmarks = results.right_hand_landmarks
                image_rows, image_cols, _ = image.shape

                # detect gestures
                if (self.last_queued is None or datetime.now() > self.last_queued + timedelta(seconds = 2)):
                    self.detect_twirl(landmarks, "R")
                    self.detect_stop_go(landmarks, "R")
                    self.detect_swipe(landmarks, "R")

                if self.curr_gesture is not None and self.curr_gesture != "":
                    self.communication_queue.put(("/gesture", self.curr_gesture))
                    self.curr_gesture = None
                    self.last_queued = datetime.now()

                cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if results.left_hand_landmarks is not None:
                landmarks = results.left_hand_landmarks
                image_rows, image_cols, _ = image.shape

                # detect gestures
                if not(self.vol_start) and (self.last_queued is None or datetime.now() > self.last_queued + timedelta(seconds = 2)):
                    self.vol_origin = None
                    self.detect_twirl(landmarks, "L")
                    self.detect_stop_go(landmarks, "L")
                    self.detect_swipe(landmarks, "L")

                if self.vol_start:
                    if self.vol_init is None:
                        print("starting x-y tracking")
                    self.vol_init = 0
                    self.detect_xy_control(landmarks, "L")
                    self.detect_twirl(landmarks, "L")

                if self.curr_gesture is not None and self.curr_gesture != "":
                    self.communication_queue.put(("/gesture", self.curr_gesture))
                    print(self.curr_gesture)
                    self.curr_gesture = None
                    self.last_queued = datetime.now()

                cv.putText(image, str(self.curr_gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
                head_x = (landmarks[self.holistic.PoseLandmark.NOSE.value].x + landmarks[self.holistic.PoseLandmark.RIGHT_EAR.value].x + landmarks[self.holistic.PoseLandmark.LEFT_EAR.value].x + landmarks[self.holistic.PoseLandmark.RIGHT_EYE.value].x + landmarks[self.holistic.PoseLandmark.LEFT_EYE.value].x) /5
                head_y = (landmarks[self.holistic.PoseLandmark.NOSE.value].y + landmarks[self.holistic.PoseLandmark.RIGHT_EAR.value].y + landmarks[self.holistic.PoseLandmark.LEFT_EAR.value].y + landmarks[self.holistic.PoseLandmark.RIGHT_EYE.value].y + landmarks[self.holistic.PoseLandmark.LEFT_EYE.value].y) /5
                head_z = (landmarks[self.holistic.PoseLandmark.NOSE.value].z + landmarks[self.holistic.PoseLandmark.RIGHT_EAR.value].z + landmarks[self.holistic.PoseLandmark.LEFT_EAR.value].z + landmarks[self.holistic.PoseLandmark.RIGHT_EYE.value].z + landmarks[self.holistic.PoseLandmark.LEFT_EYE.value].z) /5
                shoulder_x = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].x + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].x) / 2
                shoulder_y = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].y + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].y) / 2
                shoulder_z = (landmarks[self.holistic.PoseLandmark.RIGHT_SHOULDER.value].z + landmarks[
                    self.holistic.PoseLandmark.LEFT_SHOULDER.value].z) / 2

                if self.if_tracking:
                    head = {'x': head_x, 'y': head_y, 'z': head_z}
                    shoulder = {'x': shoulder_x, 'y': shoulder_y, 'z': shoulder_z}

                    smoothed_head = buffered_smooth(self.head_x, self.head_y, self.head_z, head)
                    smoothed_shoulder = buffered_smooth(self.shoulder_x, self.shoulder_y, self.shoulder_z, shoulder)

                    if smoothed_head is not None and smoothed_shoulder is not None:
                        self.communication_queue.put(("/live", [*smoothed_head, *smoothed_shoulder]))

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