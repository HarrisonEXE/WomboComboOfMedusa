import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import time

import cv2 as cv
import numpy as np
import mediapipe as mp

from Models import KeyPointClassifier
from Models import PointHistoryClassifier
from Handlers.DrawingHandler import DrawingHandler
from Helpers.CvFpsCalc import CvFpsCalc
from Helpers.Utils import isNear, euclideanDistance, variance

import math
from datetime import datetime, timedelta
from pythonosc import udp_client
from google.protobuf.json_format import MessageToDict

global client
PORT = 5005
IP = "192.168.2.2"
HISTORY_LENGTH = 16


class VisionHandler():

    def __init__(self, device=0, cap_width=960, cap_height=540, use_static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.cap = cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        self.drawingHandler = DrawingHandler(use_brect=True)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        self.keypoint_classifier_labels = self.getClassifierLabels(
            'Models/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = self.getClassifierLabels(
            'Models/point_history_classifier/point_history_classifier_label.csv')

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        self.point_history = deque(maxlen=HISTORY_LENGTH)
        self.point_history_pointer = deque(maxlen=HISTORY_LENGTH)
        self.point_history_middle = deque(maxlen=HISTORY_LENGTH)
        self.point_history_ring = deque(maxlen=HISTORY_LENGTH)

        # フィンガージェスチャー履歴 ################################################
        self.finger_gesture_history = deque(maxlen=HISTORY_LENGTH)
        self.finger_gesture_history_pointer = deque(maxlen=HISTORY_LENGTH)
        self.finger_gesture_history_middle = deque(maxlen=HISTORY_LENGTH)
        self.finger_gesture_history_ring = deque(maxlen=HISTORY_LENGTH)

    def start(self):
        self.run()
        self.cap.release()
        cv.destroyAllWindows()

    def run(self):
        mode = 0
        count_wave = 0
        count_twirl = 0
        curr_time_wave = datetime.now()
        curr_time_twirl = datetime.now()
        curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
        waving = False
        curr_time_swipe = datetime.now()
        tracker_x = []
        tracker_y = []
        tracker_z = []
        distance = []
        sent = False
        stopCount = 0

        while True:
            fps = self.cvFpsCalc.get()

            # キー処理(ESC：終了) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = self.select_mode(key, mode)

            # カメラキャプチャ #####################################################
            ret, image = self.cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # ミラー表示
            debug_image = copy.deepcopy(image)

            # 検出実施 #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # 外接矩形の計算
                    brect = self.calc_bounding_rect(
                        debug_image, hand_landmarks)
                    # ランドマークの計算
                    landmark_list = self.calc_landmark_list(
                        debug_image, hand_landmarks)

                    # 相対座標・正規化座標への変換
                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = self.pre_process_point_history(
                        debug_image, self.point_history)

                    pre_processed_landmark_list_pointer = self.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list_pointer = self.pre_process_point_history(
                        debug_image, self.point_history_pointer)

                    # pre_processed_landmark_list_middle = pre_process_landmark(landmark_list)
                    # pre_processed_point_history_list_middle = pre_process_point_history(debug_image, point_history_middle)

                    # pre_processed_landmark_list_ring = pre_process_landmark(landmark_list)
                    # pre_processed_point_history_list_ring = pre_process_point_history(debug_image, point_history_ring)

                    # 学習データ保存
                    self.logging_csv(number, mode, pre_processed_landmark_list,
                                     pre_processed_point_history_list)

                    # ハンドサイン分類
                    hand_sign_id = self.keypoint_classifier(
                        pre_processed_landmark_list)
                    hand_sign_id_pointer = self.keypoint_classifier(
                        pre_processed_landmark_list_pointer)
                    # hand_sign_id_middle = keypoint_classifier(pre_processed_landmark_list_middle)
                    # hand_sign_id_ring = keypoint_classifier(pre_processed_landmark_list_ring)

                    # just pointer finger
                    self.point_history_pointer.append(landmark_list[8])
                    # just middle finger
                    # point_history_middle.append(landmark_list[12])
                    # just ring finger
                    # point_history_ring.append(landmark_list[16])

                    # track all points
                    self.point_history.append(landmark_list[4])
                    self.point_history.append(landmark_list[8])
                    self.point_history.append(landmark_list[12])
                    self.point_history.append(landmark_list[16])
                    self.point_history.append(landmark_list[20])  # 人差指座標

                    # フィンガージェスチャー分類
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (HISTORY_LENGTH * 2):
                        finger_gesture_id = self.point_history_classifier(
                            pre_processed_point_history_list)

                    finger_gesture_id_pointer = 0
                    point_history_len_pointer = len(
                        pre_processed_point_history_list_pointer)
                    if point_history_len_pointer == (HISTORY_LENGTH * 2):
                        finger_gesture_id_pointer = self.point_history_classifier(
                            pre_processed_point_history_list_pointer)

                    # finger_gesture_id_middle = 0
                    # point_history_len_middle = len(pre_processed_point_history_list_middle)
                    # if point_history_len_middle == (history_length * 2):
                    #     finger_gesture_id_middle = point_history_classifier(pre_processed_point_history_list_middle)

                    # finger_gesture_id_ring = 0
                    # point_history_len_ring = len(pre_processed_point_history_list_ring)
                    # if point_history_len_ring == (history_length * 2):
                    #     finger_gesture_id_ring = point_history_classifier(pre_processed_point_history_list_ring)

                    # 直近検出の中で最多のジェスチャーIDを算出
                    self.finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        self.finger_gesture_history).most_common()

                    self.finger_gesture_history_pointer.append(
                        finger_gesture_id_pointer)
                    most_common_fg_id_pointer = Counter(
                        self.finger_gesture_history_pointer).most_common()

                    # self.finger_gesture_history_middle.append(finger_gesture_id_middle)
                    # most_common_fg_id_middle = Counter(self.finger_gesture_history_middle).most_common()

                    # self.finger_gesture_history_ring.append(finger_gesture_id_ring)
                    # most_common_fg_id_ring = Counter(self.finger_gesture_history_ring).most_common()

                    # 描画
                    debug_image = self.drawingHandler.draw_bounding_rect(
                        debug_image, brect)
                    debug_image = self.drawingHandler.draw_landmarks(
                        debug_image, landmark_list)
                    debug_image = self.drawingHandler.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        self.keypoint_classifier_labels[hand_sign_id],
                        self.point_history_classifier_labels[most_common_fg_id[0][0]],
                    )

                # front = detectFront(hand_landmarks, results)
                # openHand = detectOpen(hand_landmarks)
                # upright = detectUpright(hand_landmarks)

                if (datetime.now() < curr_time_wave_f and count_wave >= 20):
                    if (not(waving)):
                        client.send_message("/wave", 1)
                        print("wave detected: HI")
                    else:
                        client.send_message("/wave", 2)
                        print("wave detected: BYE")
                    waving = not(waving)
                    count_wave = 0
                    curr_time_wave = datetime.now()
                    curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
                    sent = False
                    time.sleep(5)

                elif (datetime.now() >= curr_time_wave_f):
                    curr_time_wave = datetime.now()
                    curr_time_wave_f = curr_time_wave + timedelta(seconds=5)
                    count_wave = 0

                if (self.detectOpen(hand_landmarks) and self.detectUpright(hand_landmarks) and self.detectFront(hand_landmarks, results) and (self.point_history_classifier_labels[most_common_fg_id_pointer[0][0]] == "Clockwise" or self.point_history_classifier_labels[most_common_fg_id_pointer[0][0]] == "Counter Clockwise")):
                    print(".")
                    count_wave += 1

                if waving:  # when robot is enabled

                    if (self.detectOpen(hand_landmarks) and self.detectUpright(hand_landmarks) and self.detectFront(hand_landmarks, results)):
                        stopCount += 1
                        # stop hand gesture
                        if (not(sent) and stopCount > 10):
                            # client.send_message("/stop", 0)
                            print("stop")
                            sent = True

                        # twirl hand gesture
                        curr_time_twirl = datetime.now()
                        count_twirl = 0

                    else:
                        stopCount = 0
                        sent = False

                    if (self.detectTwirlEnd(hand_landmarks, results) and curr_time_twirl + timedelta(seconds=1) > datetime.now()):
                        print("_")
                        count_twirl += 1

                    if count_twirl > 15:
                        # client.send_message("/wave", 3)
                        sent = False
                        print("twirl now")
                        count_twirl = 0
                        time.sleep(5)

                    if curr_time_swipe != None and datetime.now() > curr_time_swipe + timedelta(seconds=3):
                        curr_time_swipe = None
                        tracker_x = []
                        tracker_y = []
                        tracker_z = []

                    else:
                        if self.keypoint_classifier_labels[hand_sign_id] == "Sideways":
                            if len(tracker_x) == 0:
                                curr_time_swipe = datetime.now()
                            tracker_x.append(hand_landmarks.landmark[8].x)
                            tracker_y.append(hand_landmarks.landmark[8].y)
                            tracker_z.append(hand_landmarks.landmark[8].z)
                            distance.append(euclideanDistance(hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[
                                            12].z, hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z))

                            vari = variance(tracker_y)

                            x = np.array(tracker_x)
                            y = np.array(tracker_y)

                            a, b = np.polyfit(x, y, 1)

                            meanDistance = sum(distance)/len(distance)

                            if abs(max(tracker_x) - min(tracker_x)) >= 1.5 * meanDistance and vari < .002 and abs(a) < .15:
                                if (sum(tracker_x[0:int(len(tracker_x)/2)]) < sum(tracker_x[int(len(tracker_x)/2):])):
                                    print(len(tracker_x))
                                    print("swipe right")
                                    client.send_message("/swipe", 0)
                                else:
                                    print(len(tracker_x))
                                    print("swipe left")
                                    client.send_message("/swipe", 1)
                                sent = False
                                tracker_x = []
                                tracker_y = []
                                tracker_z = []
                                distance = []
                                vari, x, y, a, b, meanDistance, curr_time_swipe = None, None, None, None, None, None, None

            else:
                self.point_history.append([0, 0])

            debug_image = self.drawingHandler.draw_point_history(
                debug_image, self.point_history)
            debug_image = self.drawingHandler.draw_info(
                debug_image, fps, mode, number)

            # 画面反映 #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)

    def getClassifierLabels(self, path):
        with open(path, encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            labels = [row[0] for row in labels]
            return labels

    # REFACTORED
    def detectOpen(self, hand_landmarks):
        # First Finger is Open
        pseudoFixKeyPoint = hand_landmarks.landmark[6].y
        if max(hand_landmarks.landmark[7].y, hand_landmarks.landmark[8].y) < pseudoFixKeyPoint:
            return True

        # Second Finger is Open
        pseudoFixKeyPoint = hand_landmarks.landmark[10].y
        if max(hand_landmarks.landmark[11].y, hand_landmarks.landmark[12].y) < pseudoFixKeyPoint:
            return True

        # Third Finger is Open
        pseudoFixKeyPoint = hand_landmarks.landmark[14].y
        if max(hand_landmarks.landmark[15].y, hand_landmarks.landmark[16].y) < pseudoFixKeyPoint:
            return True

        # Fourth Finger is Open
        pseudoFixKeyPoint = hand_landmarks.landmark[18].y
        if max(hand_landmarks.landmark[19].y, hand_landmarks.landmark[20].y) < pseudoFixKeyPoint:
            return True

        return False

    def detectBack(self, hand_landmarks, results):
        for idx, hand_handedness in enumerate(results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            handedness = handedness_dict["classification"][0]["label"]

        if (handedness == "Right"):
            if (hand_landmarks.landmark[17].x < hand_landmarks.landmark[13].x < hand_landmarks.landmark[9].x < hand_landmarks.landmark[5].x):
                return True
        else:
            if (hand_landmarks.landmark[17].x > hand_landmarks.landmark[13].x > hand_landmarks.landmark[9].x > hand_landmarks.landmark[5].x):
                return True
        return False

    def detectTwirlEnd(self, hand_landmarks, results):
        back = detectBack(hand_landmarks, results)
        near = isNear(hand_landmarks.landmark[8], hand_landmarks.landmark[12], .06) and isNear(
            hand_landmarks.landmark[12], hand_landmarks.landmark[16], .06) and isNear(hand_landmarks.landmark[16], hand_landmarks.landmark[20], .06)
        thumbBack = hand_landmarks.landmark[4].z > hand_landmarks.landmark[8].z and hand_landmarks.landmark[
            4].z > hand_landmarks.landmark[12].z and hand_landmarks.landmark[4].z > hand_landmarks.landmark[16].z
        upright = detectUpright(hand_landmarks)

        return back and near and thumbBack and upright

    def detectUpright(self, hand_landmarks):
        pointerDistance = math.sqrt((hand_landmarks.landmark[8].x - hand_landmarks.landmark[5].x)**2 + (
            hand_landmarks.landmark[8].y - hand_landmarks.landmark[5].y)**2)
        middleDistance = math.sqrt((hand_landmarks.landmark[12].x - hand_landmarks.landmark[9].x)**2 + (
            hand_landmarks.landmark[12].y - hand_landmarks.landmark[9].y)**2)
        ringDistance = math.sqrt((hand_landmarks.landmark[16].x - hand_landmarks.landmark[13].x)**2 + (
            hand_landmarks.landmark[16].y - hand_landmarks.landmark[13].y)**2)

        pointerStraight = hand_landmarks.landmark[8].y < abs(
            hand_landmarks.landmark[5].y - pointerDistance*4/5)
        middleStraight = hand_landmarks.landmark[12].y < abs(
            hand_landmarks.landmark[9].y - middleDistance*4/5)
        ringStraight = hand_landmarks.landmark[16].y < abs(
            hand_landmarks.landmark[13].y - ringDistance*4/5)

        return pointerStraight and middleStraight and ringStraight

    def detectFront(self, hand_landmarks, results):
        for idx, hand_handedness in enumerate(results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            handedness = handedness_dict["classification"][0]["label"]

        if handedness == "Right":
            if (hand_landmarks.landmark[17].x > hand_landmarks.landmark[13].x > hand_landmarks.landmark[9].x > hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x > hand_landmarks.landmark[14].x > hand_landmarks.landmark[10].x > hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x > hand_landmarks.landmark[15].x > hand_landmarks.landmark[11].x > hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x > hand_landmarks.landmark[16].x > hand_landmarks.landmark[12].x > hand_landmarks.landmark[8].x):
                return True
        else:
            if (hand_landmarks.landmark[17].x < hand_landmarks.landmark[13].x < hand_landmarks.landmark[9].x < hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x < hand_landmarks.landmark[14].x < hand_landmarks.landmark[10].x < hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x < hand_landmarks.landmark[15].x < hand_landmarks.landmark[11].x < hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x < hand_landmarks.landmark[16].x < hand_landmarks.landmark[12].x < hand_landmarks.landmark[8].x):
                return True

        return False

    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # キーポイント
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # 相対座標に変換
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # 1次元リストに変換
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # 正規化
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # 相対座標に変換
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # 1次元リストに変換
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'Models/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'Models/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return
