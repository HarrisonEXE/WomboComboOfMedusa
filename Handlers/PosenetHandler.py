from datetime import datetime, timedelta
from queue import Queue

import cv2 as cv
import mediapipe as mp

from Handlers.DrawingHandler import DrawingHandler
from Helpers.CvFpsCalc import CvFpsCalc
from Helpers.PoseGestures import detect_hand_gesture


class PosenetHandler:
    def __init__(self, device=0, cap_width=960, cap_height=540, use_static_image_mode=True,
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

    def run(self):
        # Initialize gestures
        count_wave = 0
        curr_time_wave = datetime.now()
        curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
        waving = False

        prev_gesture = ""
        gesture = ""

        while True:
            fps = self.cvFpsCalc.get()
            key = cv.waitKey(10)

            ret, frame = self.cap.read()
            image, results = self.mediapipe_detection(frame, self.pose)

            self.drawingHandler.draw_styled_landmarks(image, results)

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
                    curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
                elif datetime.now() >= curr_time_wave_f:
                    curr_time_wave = datetime.now()
                    curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
                    count_wave = 0

                if gesture is not None:
                    if gesture != prev_gesture:
                        self.communication_queue.put(("/gesture", gesture))
                        # client.send_message("/gesture", gesture)
                    cv.putText(image, str(gesture), (1700, 140), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    prev_gesture = gesture

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
                if prev_gesture == 'wave_hello':
                    self.communication_queue.put(("/head", [head_x, head_y, head_z, shoulder_x, shoulder_y, shoulder_z]))
                    # client.send_message("/head", [head_x, head_y, head_z, shoulder_x, shoulder_y, shoulder_z])
                    print("Head: ", head_x, head_y)

            cv.putText(image, str(int(fps)) + " FPS", (10, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Posture Gesture Recognition', image)

    def mediapipe_detection(self, image, model):
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = model.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return image, results
