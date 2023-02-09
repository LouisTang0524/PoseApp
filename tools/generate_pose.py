"""
使用ViTPose对提取的关键帧检测关键点信息
生成csv标签文件用于训练
"""
import os
import warnings
import cv2
import csv
import numpy as np
from tqdm import tqdm

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def main():
    det_config = '/root/autodl-tmp/ViTPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    pose_config = '/root/autodl-tmp/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    pose_checkpoint = '/root/autodl-tmp/ViTPose/pretrained/vitpose-b.pth'

    device = 'cuda:0'
    bbox_thr = 0.3
    kpt_thr = 0.3
    det_cat_id = 1
    
    assert has_mmdet, 'Please install mmdet.'

    det_model = init_detector(
        det_config, det_checkpoint, device=device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    
    csv_label_path = 'action.csv'
    data_dir = '/root/autodl-tmp/data/RepCount/LLSP/cleaned/extracted'
    out_csv_dir = '/root/autodl-tmp/data/RepCount/LLSP/cleaned'

    for train_type in os.listdir(data_dir): # [train, test, valid]
        out_csv_path = os.path.join(out_csv_dir, train_type) + '.csv'
        with open(out_csv_path, 'w') as out_csv_file:
            csv_writer = csv.writer(out_csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            train_dir = os.path.join(data_dir, train_type)
            for action_type in os.listdir(train_dir): # [pullups, situp, ...]
                print(f"{train_type} - {action_type}")
                action_dir = os.path.join(train_dir, action_type)
                for video_name in os.listdir(action_dir): # .mp4
                    print(f"\t - {video_name}")
                    video_dir = os.path.join(action_dir, video_name)
                    for img_name in os.listdir(video_dir): # .jpg
                        img_path = os.path.join(video_dir, img_name)
                        if img_path.split(".")[-1] !="jpg":
                            continue
                        base_path = os.path.join(train_type, action_type, video_name, img_name)
                        # process each image
                        img_shape = cv2.imread(img_path).shape[:-1]
                        img_center = (img_shape[0] / 2, img_shape[1] / 2)
                        # the resulting box is (x1, y1, x2, y2)
                        mmdet_results = inference_detector(det_model, img_path)
                        # keep the person class bounding boxes.
                        person_results = process_mmdet_results(mmdet_results, det_cat_id)
                        # optional
                        return_heatmap = False

                        # e.g. use ('backbone', ) to return backbone feature
                        output_layer_names = None

                        pose_results, returned_outputs = inference_top_down_pose_model(
                            pose_model,
                            img_path,
                            person_results,
                            bbox_thr=bbox_thr,
                            format='xyxy',
                            dataset=dataset,
                            dataset_info=dataset_info,
                            return_heatmap=return_heatmap,
                            outputs=output_layer_names)
                        if len(pose_results) > 0:
                            # pose_results: array of pose_result
                            # pose_result: bbox (left,top,right,bottom) and keypoints (array[kx3], x,y,score)
                            midx = 0 # points to the bbox that locates at the center
                            dist, min_dist = 0.0, 1000000000.0
                            for idx, pose_result in enumerate(pose_results):
                                left, top, right, bottom = pose_result["bbox"][:-1]
                                bbox_center = ((top + bottom) / 2, (left + right) / 2)
                                dist = (img_center[0] - bbox_center[0]) ** 2 + (img_center[1] - bbox_center[1]) ** 2
                                if dist < min_dist:
                                    midx, min_dist = idx, dist
                                

                            pose_result = pose_results[midx]
                            keypoints = pose_result['keypoints'][:, :-1]
                            assert keypoints.shape == (17, 2), f"Unexpected landmarks shape: {keypoints.shape}"
                            csv_writer.writerow([base_path, action_type] + keypoints.flatten().astype(str).tolist())
                        else:
                            print(f"\t\t{img_name} no pose")

if __name__ == "__main__":
    main()