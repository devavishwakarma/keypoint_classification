import cv2
import numpy as np
import copy
import itertools


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, np.array([[x, y]]), axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    if len(temp_landmark_list) > 0:
        base_x, base_y = temp_landmark_list[0]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(point_history):
    # Convert all (x, y) tuples to [x, y] lists
    temp_point_history = [list(p) for p in point_history]

    base_x, base_y = temp_point_history[0]

    for index in range(len(temp_point_history)):
        temp_point_history[index][0] = temp_point_history[index][0] - base_x
        temp_point_history[index][1] = temp_point_history[index][1] - base_y

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    max_value = max(list(map(abs, temp_point_history)))

    if max_value == 0:
        max_value = 1

    normalized_point_history = [value / max_value for value in temp_point_history]
    return normalized_point_history
