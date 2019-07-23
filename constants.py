import os

def model_save_dir():
    return os.path.join(os.path.dirname(__file__), "saved_models")

def get_simulator_path():
    return os.path.join(os.path.dirname(__file__), "CarND-Behavioral-Cloning-P3", "drive.py")