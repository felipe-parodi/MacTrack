import os
import re

import numpy as np
import yaml
import yaml.constructor

CAMERA_RESOLUTION = [1920, 1080]
CAMERA_TYPE = "rgb"


def convert_yaml_to_json(yaml_file):
    """Convert a yaml calibration file to a json calibration file

    Args:
        yaml_file: yaml calibration file

    Returns:
        json_file: json calibration file
    """
    with open(yaml_file, "r") as f:
        # Skip the first line of the file
        f.readline()
        yaml_str = f.read()

        # Remove the opencv-matrix tag
        yaml_str = yaml_str.replace("!!opencv-matrix", "")

        data = yaml.safe_load(yaml_str)
    K = np.array(data["intrinsicMatrix"]["data"]).reshape((3, 3), order="F")
    distCoef = np.array(data["distortionCoefficients"]["data"]).reshape(
        (1, 5), order="F"
    )
    rotation = np.array(data["R"]["data"]).reshape((3, 3), order="F")
    translation = np.array(data["T"]["data"]).reshape((3, 1), order="F")
    json_data = {}
    json_data["name"] = os.path.basename(yaml_file).split(".")[0]
    json_data["type"] = CAMERA_TYPE
    json_data["resolution"] = CAMERA_RESOLUTION
    json_data["K"] = K.tolist()
    json_data["distCoef"] = distCoef.tolist()
    json_data["R"] = rotation.tolist()
    json_data["T"] = translation.tolist()

    return json_data


def create_input_list(video_directory):
    """Create input list for merge from a directory of AVI video files.

    Args:
        video_directory (str): Path to directory containing AVI video files.

    Returns:
        None
    """
    video_files = [f for f in os.listdir(video_directory) if f.endswith(".avi")]

    pairs = {}

    for video_file in video_files:
        key = video_file[:7]
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(video_file)

    for key, video_pair in pairs.items():
        if len(video_pair) == 2:
            input_filename = f"input_videos_{key}.txt"
            # save to video directory:
            with open(os.path.join(video_directory, input_filename), "w") as f:
                for video in video_pair:
                    f.write(f"file '{video}'\n")
