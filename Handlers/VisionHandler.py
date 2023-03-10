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

from utils import CvFpsCalc
from Models import KeyPointClassifier
from Models import PointHistoryClassifier

import math
from datetime import datetime, timedelta
from pythonosc import udp_client
from google.protobuf.json_format import MessageToDict

global client
PORT = 5005
IP = "192.168.2.2"


class VisionHandler():

    def __init__(self):
        pass

    def start(self):
        # 引数解析 #################################################################
        args = self.get_args()

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        use_brect = True

        # カメラ準備 ###############################################################
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # モデルロード #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        # ラベル読み込み ###########################################################
        with open('Models/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'Models/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS計測モジュール ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # 座標履歴 #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)
        point_history_pointer = deque(maxlen=history_length)
        point_history_middle = deque(maxlen=history_length)
        point_history_ring = deque(maxlen=history_length)

        # フィンガージェスチャー履歴 ################################################
        finger_gesture_history = deque(maxlen=history_length)
        finger_gesture_history_pointer = deque(maxlen=history_length)
        finger_gesture_history_middle = deque(maxlen=history_length)
        finger_gesture_history_ring = deque(maxlen=history_length)

        #  ########################################################################
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
            fps = cvFpsCalc.get()

            # キー処理(ESC：終了) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = self.select_mode(key, mode)

            # カメラキャプチャ #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # ミラー表示
            debug_image = copy.deepcopy(image)

            # 検出実施 #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
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
                        debug_image, point_history)

                    pre_processed_landmark_list_pointer = self.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list_pointer = self.pre_process_point_history(
                        debug_image, point_history_pointer)

                    # pre_processed_landmark_list_middle = pre_process_landmark(landmark_list)
                    # pre_processed_point_history_list_middle = pre_process_point_history(debug_image, point_history_middle)

                    # pre_processed_landmark_list_ring = pre_process_landmark(landmark_list)
                    # pre_processed_point_history_list_ring = pre_process_point_history(debug_image, point_history_ring)

                    # 学習データ保存
                    self.logging_csv(number, mode, pre_processed_landmark_list,
                                     pre_processed_point_history_list)

                    # ハンドサイン分類
                    hand_sign_id = keypoint_classifier(
                        pre_processed_landmark_list)
                    hand_sign_id_pointer = keypoint_classifier(
                        pre_processed_landmark_list_pointer)
                    # hand_sign_id_middle = keypoint_classifier(pre_processed_landmark_list_middle)
                    # hand_sign_id_ring = keypoint_classifier(pre_processed_landmark_list_ring)

                    # just pointer finger
                    point_history_pointer.append(landmark_list[8])
                    # just middle finger
                    # point_history_middle.append(landmark_list[12])
                    # just ring finger
                    # point_history_ring.append(landmark_list[16])

                    # track all points
                    point_history.append(landmark_list[4])
                    point_history.append(landmark_list[8])
                    point_history.append(landmark_list[12])
                    point_history.append(landmark_list[16])
                    point_history.append(landmark_list[20])  # 人差指座標

                    # フィンガージェスチャー分類
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    finger_gesture_id_pointer = 0
                    point_history_len_pointer = len(
                        pre_processed_point_history_list_pointer)
                    if point_history_len_pointer == (history_length * 2):
                        finger_gesture_id_pointer = point_history_classifier(
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
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()

                    finger_gesture_history_pointer.append(
                        finger_gesture_id_pointer)
                    most_common_fg_id_pointer = Counter(
                        finger_gesture_history_pointer).most_common()

                    # finger_gesture_history_middle.append(finger_gesture_id_middle)
                    # most_common_fg_id_middle = Counter(finger_gesture_history_middle).most_common()

                    # finger_gesture_history_ring.append(finger_gesture_id_ring)
                    # most_common_fg_id_ring = Counter(finger_gesture_history_ring).most_common()

                    # 描画
                    debug_image = self.draw_bounding_rect(
                        use_brect, debug_image, brect)
                    debug_image = self.draw_landmarks(
                        debug_image, landmark_list)
                    debug_image = self.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
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

                if (self.detectOpen(hand_landmarks) and self.detectUpright(hand_landmarks) and self.detectFront(hand_landmarks, results) and (point_history_classifier_labels[most_common_fg_id_pointer[0][0]] == "Clockwise" or point_history_classifier_labels[most_common_fg_id_pointer[0][0]] == "Counter Clockwise")):
                    print(".")
                    count_wave += 1

                if waving:  # when robot is enabled

                    if (self.detectOpen(hand_landmarks) and self.detectUpright(hand_landmarks) and self.detectFront(hand_landmarks, results)):
                        stopCount += 1
                        # stop hand gesture
                        if (not(sent) and stopCount > 10):
                            client.send_message("/stop", 0)
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
                        client.send_message("/wave", 3)
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
                        if keypoint_classifier_labels[hand_sign_id] == "Sideways":
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
                point_history.append([0, 0])

            debug_image = self.draw_point_history(debug_image, point_history)
            debug_image = self.draw_info(debug_image, fps, mode, number)

            # 画面反映 #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)

        cap.release()
        cv.destroyAllWindows()

    # TODO: ensure functionality is retained
    def detectOpen(self, hand_landmarks):
        start_time = time.perf_counter()

        firstFingerIsOpen = False
        secondFingerIsOpen = False
        thirdFingerIsOpen = False
        fourthFingerIsOpen = False

        pseudoFixKeyPoint = hand_landmarks.landmark[6].y
        firstFingerIsOpen = hand_landmarks.landmark[
            7].y < pseudoFixKeyPoint and hand_landmarks.landmark[8].y < pseudoFixKeyPoint

        pseudoFixKeyPoint = hand_landmarks.landmark[10].y
        secondFingerIsOpen = hand_landmarks.landmark[
            11].y < pseudoFixKeyPoint and hand_landmarks.landmark[12].y < pseudoFixKeyPoint

        pseudoFixKeyPoint = hand_landmarks.landmark[14].y
        thirdFingerIsOpen = hand_landmarks.landmark[
            15].y < pseudoFixKeyPoint and hand_landmarks.landmark[16].y < pseudoFixKeyPoint

        pseudoFixKeyPoint = hand_landmarks.landmark[18].y
        fourthFingerIsOpen = hand_landmarks.landmark[
            19].y < pseudoFixKeyPoint and hand_landmarks.landmark[20].y < pseudoFixKeyPoint

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The execution time is: {execution_time}")

        if not(firstFingerIsOpen or secondFingerIsOpen or thirdFingerIsOpen or fourthFingerIsOpen):
            return False
        else:
            return True

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=960)
        parser.add_argument("--height", help='cap height',
                            type=int, default=540)

        parser.add_argument('--use_static_image_mode', action='store_true')
        parser.add_argument("--min_detection_confidence",
                            help='min_detection_confidence',
                            type=float,
                            default=0.7)
        parser.add_argument("--min_tracking_confidence",
                            help='min_tracking_confidence',
                            type=int,
                            default=0.5)

        args = parser.parse_args()

        return args

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

        if (back and near and thumbBack and upright):
            return True

    def euclideanDistance(self, a_x, a_y, a_z, b_x, b_y, b_z):
        return math.sqrt(math.pow((a_x-b_x), 2) + math.pow((a_y-b_y), 2) + math.pow((a_z-b_z), 2))

    def isNear(self, fingerOne, fingerTwo, threshold):
        return euclideanDistance(fingerOne.x, fingerOne.y, fingerOne.z, fingerTwo.x, fingerTwo.y, fingerTwo.z) < threshold

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

        if (pointerStraight and middleStraight and ringStraight):
            return True
        else:
            return False

    def detectFront(self, hand_landmarks, results):
        for idx, hand_handedness in enumerate(results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            handedness = handedness_dict["classification"][0]["label"]

        if (handedness == "Right"):
            if (hand_landmarks.landmark[17].x > hand_landmarks.landmark[13].x > hand_landmarks.landmark[9].x > hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x > hand_landmarks.landmark[14].x > hand_landmarks.landmark[10].x > hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x > hand_landmarks.landmark[15].x > hand_landmarks.landmark[11].x > hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x > hand_landmarks.landmark[16].x > hand_landmarks.landmark[12].x > hand_landmarks.landmark[8].x):
                return True
        else:
            if (hand_landmarks.landmark[17].x < hand_landmarks.landmark[13].x < hand_landmarks.landmark[9].x < hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x < hand_landmarks.landmark[14].x < hand_landmarks.landmark[10].x < hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x < hand_landmarks.landmark[15].x < hand_landmarks.landmark[11].x < hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x < hand_landmarks.landmark[16].x < hand_landmarks.landmark[12].x < hand_landmarks.landmark[8].x):
                return True
        return False

    def variance(self, data_y):
        # Number of observations
        n = len(data_y)
        # Mean of the data
        mean = sum(data_y) / n
        # Square deviations
        deviations = [(y - mean) ** 2 for y in data_y]
        # Variance
        variance = sum(deviations) / n
        return variance

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
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def draw_landmarks(self, image, landmark_point):
        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # 人差指
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # 中指
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # 薬指
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # 小指
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # 手の平
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # キーポイント
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # 外接矩形
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text,
                       finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)

        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                          (152, 251, 152), 2)

        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
        return image
