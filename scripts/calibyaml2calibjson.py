# Author: Felipe Parodi
# Date: 2023-03-28
# Description: Convert a yaml calibration file to a json calibration file
# Project: MacTrack

import argparse
import json
import pathlib

import yaml
import yaml.constructor

from helperfunctions_mt import OpenCVMatrixConstructor, convert_yaml_to_json

CAMERA_RESOLUTION = [1920, 1080]
CAMERA_TYPE = "rgb"

def main():
    """Convert multiple YAML calibration files to one JSON calibration file."""
    parser = argparse.ArgumentParser(
        description="Convert multiple YAML calibration files to one JSON calibration file."
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the input directory containing YAML calibration files.",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default="calibration_output.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    calibration_files = list(args.input_dir.glob("*.yaml"))
    output_dict = {}
    output_dict["cameras"] = []

    print(
        f"There are {len(calibration_files)} calibration files in the input directory."
    )

    for yaml_file in calibration_files:
        json_data = convert_yaml_to_json(yaml_file)
        output_dict["cameras"].append(json_data)

    output_dict["calibDataSource"] = "230210_3_mactrack"

    with open(args.output_file, "w") as f:
        json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
