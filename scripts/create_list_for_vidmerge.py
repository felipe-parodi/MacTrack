# Author: Felipe Parodi
# Date: 2023-03-27
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
import os


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create input list for merge")
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="path/to/video/directory",
        help="Path to video directory",
    )
    args = parser.parse_args()

    video_directory = args.input_dir
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


if __name__ == "__main__":
    main()
