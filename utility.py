import os
import constants
import csv
import os
import cv2
import numpy as np
import subprocess
import sys

def parse_drive_log(csv_file_paths):
    images = []
    measurements = []

    for csv_file_path in csv_file_paths:
        lines = []

        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip the headers
            for line in reader:
                lines.append(line)



        for line in lines:
            source_path = line[0]
            left_source_path = line[1]
            right_source_path = line[2]
            current_path = os.path.join(os.path.dirname(csv_file_path), source_path)
            left_camera_path = os.path.join(os.path.dirname(csv_file_path), left_source_path)
            right_camera_path = os.path.join(os.path.dirname(csv_file_path), right_source_path)

            correction = 0.2
            image = cv2.imread(current_path)
            left_image = cv2.imread(left_camera_path)
            right_image = cv2.imread(right_camera_path)

            images.append(image)
            measurements.append(float(line[3]))

            images.append(cv2.flip(image, 1))
            measurements.append(float(line[3]) * -1.0)
            # images.append(left_image)
            # measurements.append(float(line[3]) + correction)
            #
            # images.append(right_image)
            # measurements.append(float(line[3]) - correction)
            # measurement = {}
            # measurement["steer"] = float(line[3])
            # measurement["throttle"] = float(line[4])
            # measurement["brake"] = float(line[5])
            # measurement["speed"] = float(line[6])
    print (images.shape, measurements.shape)
    return np.array(images), np.array(measurements)

def model_save_path(name):
    return os.path.join(constants.model_save_dir(), name)
