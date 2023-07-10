# Author: Felipe Parodi
# Date: 2023-07-09
# Project: MacTrack
# Description: This script converts a directory of videos to a COCO json file with detection labels.
# Usage: python mtrack_viddir2coco_3x.py --input-dir /path/to/images --output-dir /path/to/output

import argparse
import json
import os
import time

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

start_time = time.time()

def save_checkpoint(img_anno_dict, out_json_file, checkpoint_count):
    checkpoint_file = out_json_file.replace(
        ".json", f"_checkpoint_{checkpoint_count}.json"
    )
    with open(checkpoint_file, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input-dir", type=str, help="path to videos")
    parser.add_argument("--output-dir", type=str, help="path to the output directory")
    parser.add_argument("--bbox-thr", default=0.95, type=float, help="bbox threshold")
    parser.add_argument(
        "--checkpoint-interval", default=5, type=int, help="checkpoint interval"
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="device to use")
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="The transparency of bboxes"
    )
    parser.add_argument(
        "--draw-bbox", action="store_true", help="Draw bboxes of instances"
    )
    args = parser.parse_args()

    vid_dir = args.input_dir
    out_dir = args.output_dir
    bbox_thr = args.bbox_thr
    checkpoint_interval = args.checkpoint_interval
    device = args.device
    keypoint_thr = 0.5
    min_num_keypoints_desired = 11

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "/imgs", exist_ok=True)
    os.makedirs(out_dir + "/viz", exist_ok=True)
    os.makedirs(out_dir + "/annotations", exist_ok=True)

    out_json_file = out_dir + "/annotations/enclosure_pose_labels.json"

    det_config = "C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\scripts\\pose_pseudolabel_230612\\fasterrcnn_2classdet_mt_3x.py"
    det_checkpoint = "Y:\\MacTrack\\results\\mactrack_detection\\fasterrcnn2class_best_bbox_mAP_epoch_50.pth"
    pose_config = "C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\scripts\\pose_pseudolabel_230612\\hrnet_w48_macaque_256x192_3x.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth"

    print(f"Output json file: {out_json_file}. \a\n Initializing NHP detection and pose estimation models ...")
    det_model = init_detector(det_config, det_checkpoint, device=device)
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
    pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device)
    pose_estimator.cfg.visualizer.radius = 4
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 2
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=pose_estimator.cfg.skeleton_style
    )
    categories = [{"id": 1, "name": "monkey", "supercategory": "monkey"}]

    img_anno_dict = {
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    video_list = os.listdir(vid_dir)
    print(f"Successfully initialized models. Now loading {len(video_list)} videos ...")
    uniq_id_list = []
    frame_id_uniq_counter = 0
    ann_uniq_id, checkpoint_count = int(0), int(0)
    id_pool = np.arange(0, 200_000)
    np.random.shuffle(id_pool)
    for vid_idx, vid in enumerate(video_list):
        if not vid.endswith((".mp4", ".avi")):
            continue
        video = mmcv.VideoReader(vid_dir + vid)
        print(vid)

        for frame_id, cur_frame in enumerate(video):

            detection_results = inference_detector(det_model, cur_frame)

            if len(detection_results.pred_instances) == 0:
                continue
            if len(detection_results.pred_instances) > 2:
                detection_results.pred_instances = detection_results.pred_instances[:2]
            pred_instance = detection_results.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.9)
            ]
            bboxes = bboxes[nms(bboxes, 0.3), :4]

            # predict keypoints
            pose_results = inference_topdown(pose_estimator, cur_frame, bboxes)
            if len(pose_results) == 0:
                continue
            data_samples = merge_data_samples(pose_results)

            visualizer.add_datasample(
                "result",
                cur_frame,
                data_sample=data_samples,
                draw_gt=False,
                kpt_thr=keypoint_thr,
            )

            frame = visualizer.get_image()
            height, width, _ = cur_frame.shape
            bboxes = detection_results.pred_instances.bboxes.cpu().numpy()
            scores = detection_results.pred_instances.scores.cpu().numpy()
            labels = detection_results.pred_instances.labels.cpu().numpy()

            keypoints = data_samples.pred_instances.keypoints
            keypoint_scores = data_samples.pred_instances.keypoint_scores
            
            annotations_added = False
            visible_keypoints = 0
            annotations_list = []

            for bbox, score, label, kpts, kpts_scores in zip(
                bboxes, scores, labels, keypoints, keypoint_scores
            ):
                visible_keypoints = 0
                if score < bbox_thr:
                    continue
                bbox_top_left_x, bbox_top_left_y = bbox[0], bbox[1]
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]

                bbox = [bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height]
                if (
                    bbox_width == 0
                    or bbox_height == 0
                    or bbox_width < 0
                    or bbox_height < 0
                ):
                    continue
                elif bbox_top_left_x < 0 or bbox_top_left_y < 0:
                    continue
                elif (
                    bbox_top_left_x + bbox_width > width
                    or bbox_top_left_y + bbox_height > height
                ):
                    continue

                area = round(bbox_width * bbox_height, 2)
                center = [
                    bbox_top_left_x + bbox_width / 2,
                    bbox_top_left_y + bbox_height / 2,
                ]
                scale = [bbox_width / 200, bbox_height / 200]

                frame_id_uniq = id_pool[frame_id_uniq_counter]
                frame_id_uniq_counter += 1
                kpts_flat = []
                for pt, pt_score in zip(kpts, kpts_scores):
                    if pt_score > keypoint_thr:
                        visibility = 2  # visible
                        visible_keypoints += 1
                    else:
                        visibility = 0  # not marked

                    kpts_flat.extend([float(pt[0]), float(pt[1]), visibility])
                uniq_id_list.append(frame_id_uniq)
                file_name = (
                    os.path.basename(vid)[:-4] + "_" + str(frame_id_uniq) + ".jpg"
                )

                images = {
                    "file_name": file_name,
                    "height": height,
                    "width": width,
                    "id": frame_id_uniq,
                }

                if visible_keypoints >= min_num_keypoints_desired:
                    annotations = {
                        "keypoints": kpts_flat,
                        "num_keypoints": visible_keypoints,
                        "area": float(area),
                        "iscrowd": 0,
                        "image_id": int(frame_id_uniq),
                        "bbox": [float(i) for i in bbox],
                        "center": [float(i) for i in center],
                        "scale": [float(i) for i in scale],
                        "category_id": 1,
                        "id": int(ann_uniq_id),
                    }
                    annotations_list.append(annotations)

            if len(annotations_list) == len(bboxes):
                for annotation in annotations_list:
                    img_anno_dict['annotations'].append(annotation)
                    ann_uniq_id += 1
                raw_frame = out_dir + "/imgs/" + file_name
                cv2.imwrite(raw_frame, cur_frame)
                viz_frame = out_dir + "/viz/" + file_name[:-4] + "_vis.jpg"
                cv2.imwrite(viz_frame, frame)
                annotations_added = True        
            visible_keypoints = 0
            if annotations_added:
                img_anno_dict['images'].append(images)

        if (vid_idx + 1) % checkpoint_interval == 0:
            checkpoint_count += 1
            print(f"Checkpointing at video: {vid_idx+1}")
            save_checkpoint(img_anno_dict, out_json_file, checkpoint_count)

    print(f"Number of images added to COCO json: {len(img_anno_dict['images'])}")

    with open(out_json_file, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=2)

    print("Time elapsed: ", time.time() - start_time)

if __name__ == "__main__":
    main()
