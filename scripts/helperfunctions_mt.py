import os
import re

import yaml
import yaml.constructor

class OpenCVMatrixConstructor(yaml.constructor.SafeConstructor):
    def construct_opencv_matrix(self, node):
        # Parse the YAML node into a Python dict
        data = self.construct_yaml_map(node)

        # Extract the matrix dimensions and data type
        rows = int(data["rows"])
        cols = int(data["cols"])
        dtype = data["dt"]

        # Extract the matrix data as a flat list of strings
        data_str = data["data"].replace("\n", " ")
        data_list = re.split(r"\s+", data_str.strip())

        # Convert the data to the appropriate data type
        if dtype == "d":
            data = [float(x) for x in data_list]
        elif dtype == "f":
            data = [float(x) for x in data_list]
        elif dtype == "i":
            data = [int(x) for x in data_list]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Reshape the data into a matrix
        matrix = [data[i : i + cols] for i in range(0, len(data), cols)]

        return matrix


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

        # Parse the YAML data using the custom constructor
        loader = yaml.SafeLoader(yaml_str)
        loader.add_constructor(
            "!<tag:yaml.org,2002:opencv-matrix>",
            OpenCVMatrixConstructor.construct_opencv_matrix,
        )
        data = loader.get_single_data()

    # Extract the matrix dimensions and data type
    rows = int(data["intrinsicMatrix"]["rows"])
    cols = int(data["intrinsicMatrix"]["cols"])
    dtype = data["intrinsicMatrix"]["dt"]

    # Extract the matrix data as a list of lists
    matrix_data = [float(x) for x in data["intrinsicMatrix"]["data"]]
    matrix = [matrix_data[i : i + cols] for i in range(0, len(matrix_data), cols)]

    # Convert the YAML data to a JSON dictionary
    json_data = {}
    json_data["name"] = os.path.basename(yaml_file).split(".")[0]
    json_data["type"] = CAMERA_TYPE
    json_data["resolution"] = CAMERA_RESOLUTION
    json_data["K"] = matrix
    json_data["distCoef"] = data["distortionCoefficients"]["data"]
    # json_data["R"] = data["R"]["data"]
    json_data["R"] = [
        [data["R"]["data"][0], data["R"]["data"][1], data["R"]["data"][2]],
        [data["R"]["data"][3], data["R"]["data"][4], data["R"]["data"][5]],
        [data["R"]["data"][6], data["R"]["data"][7], data["R"]["data"][8]],
    ]
    json_data["t"] = data["T"]["data"]

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
