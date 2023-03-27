import numpy as np
import cv2
import math

hand_landmarks = {
    'thumb': [1, 2, 3, 4],
    'indexFinger': [5, 6, 7, 8, 9],
    'middleFinger': [9, 10, 11, 12],
    'thirdFinger': [13, 14, 15, 16],
    'pinkieFinger': [17, 18, 19, 20],
    'palm': [0, 5, 9, 13, 17]
}


def angle(self, point1, point2, point3):
    """ Calculate angle between two lines """
    if point1 == (0, 0) or point2 == (0, 0) or point3 == (0, 0):
        return 0

    numerator = point2[1] * (point1[0] - point3[0]) + point1[1] * (point3[0] - point2[0]) + point3[1] * (
            point2[0] - point1[0])

    denominator = (point2[0] - point1[0]) * (point1[0] - point3[0]) + (point2[1] - point1[1]) * (point1[1] - point3[1])

    try:
        ang = math.atan(numerator / denominator)
        ang = ang * 180 / math.pi
        if ang < 0:
            ang = 180 + ang
        return ang
    except:
        return 90.0


def angle_of_singleline(point1, point2):
    """ Calculate angle of a single line """
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]

    return math.degrees(math.atan2(y_diff / x_diff))


def dist_xy(point1, point2):
    """ Euclidean distance between two points """
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def point_position(self, point, line_pt_1, line_pt_2):
    """
    Left or Right position of the point from a line
    """
    value = (line_pt_2[0] - line_pt_1[0]) * (point[1] - line_pt_1[1]) - (line_pt_2[1] - line_pt_1[1]) * (
            point[0] - line_pt_1[0])

    if value >= 0:
        return "left"

    return "right"


def calc_bounding_rect(image, landmarks):
    """ Calculate bounding rectangle of the hand """
    image_width, image_height = image.shape[1], image.shape[0]

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)

        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def detect_hand_gesture(landmarks, handedness):
    """ Detect hand gesture """
    if handedness == "R":
        movements = {
            'close': hand_close(landmarks),
            'upright': hand_upright(landmarks),
            'front': not hand_front(landmarks),
            'back': not hand_back(landmarks)
        }
    elif handedness == "L":
        movements = {
            'close': hand_close(landmarks),
            'upright': hand_upright(landmarks),
            'front': hand_front(landmarks),
            'back': hand_back(landmarks)
        }
    elif handedness == "Both":
        movements = {
            'clap' : hand_clap(landmarks)
        }
    # print(movements)
    return movements

def hand_clap(landmarks):
    """ Detect hand clap """
    left_wrist = [landmarks.left_hand_landmarks.landmark[0].x, landmarks.left_hand_landmarks.landmark[0].y]
    right_write = [landmarks.right_hand_landmarks.landmark[0].x, landmarks.right_hand_landmarks.landmark[0].y]
    distance = dist_xy(left_wrist, right_write)
    if distance < 0.1:
        return True
    else:
        return False

def hand_close(landmarks):
    """ Extract the x and y coordinates of the wrist landmarks """
    point = landmarks.landmark[6].y
    first_finger_is_open = landmarks.landmark[7].y < point and landmarks.landmark[8].y < point

    point = landmarks.landmark[10].y
    second_finger_is_open = landmarks.landmark[11].y < point and landmarks.landmark[12].y < point

    point = landmarks.landmark[14].y
    third_finger_is_open = landmarks.landmark[15].y < point and landmarks.landmark[16].y < point

    point = landmarks.landmark[18].y
    fourth_finger_is_open = landmarks.landmark[19].y < point and landmarks.landmark[20].y < point

    if not (first_finger_is_open or second_finger_is_open or third_finger_is_open or fourth_finger_is_open):
        return True
    else:
        return False


def hand_upright(landmarks):
    pointer_distance = math.sqrt((landmarks.landmark[8].x - landmarks.landmark[5].x) ** 2 + (
            landmarks.landmark[8].y - landmarks.landmark[5].y) ** 2)
    middle_distance = math.sqrt((landmarks.landmark[12].x - landmarks.landmark[9].x) ** 2 + (
            landmarks.landmark[12].y - landmarks.landmark[9].y) ** 2)
    ring_distance = math.sqrt((landmarks.landmark[16].x - landmarks.landmark[13].x) ** 2 + (
            landmarks.landmark[16].y - landmarks.landmark[13].y) ** 2)

    pointer_straight = landmarks.landmark[8].y < abs(landmarks.landmark[5].y - pointer_distance * 4 / 5)
    middle_straight = landmarks.landmark[12].y < abs(landmarks.landmark[9].y - middle_distance * 4 / 5)
    ring_straight = landmarks.landmark[16].y < abs(landmarks.landmark[13].y - ring_distance * 4 / 5)

    if pointer_straight and middle_straight and ring_straight:
        return True
    else:
        return False


def hand_front(landmarks):
    if (landmarks.landmark[17].x > landmarks.landmark[13].x > landmarks.landmark[9].x >
        landmarks.landmark[5].x) and (
            landmarks.landmark[18].x > landmarks.landmark[14].x > landmarks.landmark[10].x >
            landmarks.landmark[6].x) and landmarks.landmark[19].x > landmarks.landmark[15].x > \
            landmarks.landmark[11].x > landmarks.landmark[7].x and landmarks.landmark[20].x > \
            landmarks.landmark[16].x > landmarks.landmark[12].x > landmarks.landmark[8].x:
        return True
    else:
        return False


def hand_back(landmarks):
    if landmarks.landmark[17].x < landmarks.landmark[13].x < landmarks.landmark[9].x < landmarks.landmark[5].x:
        return True
    else:
        return False


def hand_twirl_end(landmarks):
    if hand_back(landmarks) and (
            is_near(landmarks.landmark[8], landmarks.landmark[12]) and is_near(landmarks.landmark[12],
                                                                               landmarks.landmark[16]) and is_near(
        landmarks.landmark[16], landmarks.landmark[20])) and (
            landmarks.landmark[4].z > landmarks.landmark[8].z and landmarks.landmark[4].z > landmarks.landmark[12].z and
            landmarks.landmark[4].z > landmarks.landmark[16].z):
        return True


def is_near(finger_one, finger_two):
    return dist_xy([finger_one.x, finger_one.y], [finger_two.x, finger_two.y]) < .05

def detect_pose(landmarks):
    """ Detect pose """
    movements = {
        'bow': pose_bow(landmarks)
    }
    # print(movements)
    return movements

def pose_bow(landmarks):
    nose_y = landmarks.landmark[0].y
    left_shoulder_y = landmarks.landmark[11].y
    right_shoulder_y = landmarks.landmark[12].y

    # Calculate the difference between nose y-coordinate and the average of the left and right shoulder y-coordinates
    shoulder_y_avg = (left_shoulder_y + right_shoulder_y) / 2
    diff = nose_y - shoulder_y_avg

    # Check if the head is bowed down
    if diff > 0.5:  # tweak this threshold to suit your needs
        # print("Head bowed down!")
        return True
    else:
        return False