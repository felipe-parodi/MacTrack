# Author: Felipe Parodi
# Date: 2023-03-28
# Project: MacTrack
# Description: Create input list for merging videos

"""
Usage: python create_list_for_vidmerge.py --input-dir Y:\CameraCalibration\230201_30by30\
After this, run the following command in the same directory as the input list:
ffmpeg -f concat -safe 0 -i input_videos_e3v822f.txt -c copy stitched-vids/e3v822f.avi
Then, cut the videos to the exact frame count.
later: 1. write loop for above;
"""

import argparse

from helperfunctions_mt import create_input_list


def main():
    parser = argparse.ArgumentParser(description="Create input list for merge")
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="path/to/video/directory",
        help="Path to video directory",
    )
    args = parser.parse_args()

    create_input_list(args.input_dir)


if __name__ == "__main__":
    main()
