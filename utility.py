import os
import constants
import csv
import os
import cv2
import numpy as np
import subprocess
import sys

def parse_drive_log(csv_file_path):
    lines = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    for line in lines:
        source_path = line[0]
        current_path = os.path.join(os.path.dirname(csv_file_path), source_path)
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))
        # measurement = {}
        # measurement["steer"] = float(line[3])
        # measurement["throttle"] = float(line[4])
        # measurement["brake"] = float(line[5])
        # measurement["speed"] = float(line[6])

    return np.array(images), np.array(measurements)

def model_save_path(name):
    return os.path.join(constants.model_save_dir(), name)
