"""
Evaluate the performance of the model.
"""
import pandas as pd
import numpy as np
import os
import warnings
import cv2
import csv
from sklearn.model_selection import train_test_split
import torch

from models.mlp_light import MLP
from tools.counter import RepetitionCounter

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:,:,0], axis = 1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:,:,0], axis = 1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:,:,1], axis = 1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:,:,1], axis = 1), 1)

    all_landmarks[:,:,0] = (all_landmarks[:,:,0] - x_min) / (x_max - x_min)
    all_landmarks[:,:,1] = (all_landmarks[:,:,1] - y_min) / (y_max - y_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks


def obtain_landmarks_label(csv_path, all_landmarks, all_labels, label2index):
    file_seperator = ','
    n_landmarks = 17
    n_dimensions = 3
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_seperator)
        for row in csv_reader:
            assert len(row) == n_landmarks * n_dimensions + 2, f'wrong number of values: {len(row)}'
            landmarks = np.array(row[2:], np.float32).reshape([n_landmarks, n_dimensions])
            all_landmarks.append(landmarks)
            label = label2index[row[1]]
            all_labels.append(label)
    return all_landmarks, all_labels


def csv2data(train_csv, action2index):
    train_landmarks = []
    train_labels = []
    train_landmarks, train_labels = obtain_landmarks_label(train_csv, train_landmarks, train_labels, action2index)

    train_landmarks = np.array(train_landmarks)
    train_labels = np.array(train_labels)
    train_landmarks = normalize_landmarks(train_landmarks)

    return train_landmarks, train_labels


def config_detection():
    """
    return det_model, pose_model, dataset, dataset_info
    """
    det_config = '/root/autodl-tmp/ViTPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    pose_config = '/root/autodl-tmp/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
    pose_checkpoint = '/root/autodl-tmp/ViTPose/pretrained/vitpose-b.pth'
    device = 'cuda:0'
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

    return det_model, pose_model, dataset, dataset_info


if __name__ == '__main__':

    bbox_thr = 0.3
    kpt_thr = 0.3
    det_cat_id = 1
    radius = 4
    thickness = 1
    det_model, pose_model, dataset, dataset_info = config_detection()
    
    csv_label_path = 'action.csv'
    root_dir = '/root/autodl-tmp/data/RepCount/LLSP'
    # output_video_dir = os.path.join(root_dir, 'video_visual_output/test')
    input_video_dir = os.path.join(root_dir, 'video/test')
    label_dir = os.path.join(root_dir, 'annotation')
    label_filename = os.path.join(label_dir, 'test.csv')

    data_root = os.path.join(root_dir, 'extracted')
    test_csv = os.path.join(root_dir, 'test.csv')

    label_pd = pd.read_csv(csv_label_path)
    index_label_dict = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index_label_dict[label] = action
    
    num_classes = len(index_label_dict)
    
    action2index = {v: k for k, v in index_label_dict.items()}

    model = MLP(None, None, 1e-3, seed=42, num_classes=num_classes)
    weight_path = 'best_weights.pth'
    new_weights = torch.load(weight_path, map_location='cuda:0')
    model.layers.load_state_dict(new_weights)
    model.eval()

    df = pd.read_csv(label_filename)
    testMAE = []
    testOBO = []
    for i in range(len(df)):
        filename = df.loc[i, 'name']
        gt_count = df.loc[i, 'count']
        video_path = os.path.join(input_video_dir, filename)
        # output_video_path = os.path.join(output_video_dir, filename)
        print('video input path', video_path)
        #print('video output path', output_video_path)
        video_cap = cv2.VideoCapture(video_path)

        # get some video parameters to generate output video with classification
        video_n_frame = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

        frame_idx = 0
        output_frame = None

        action_counts = [0] * num_classes

        while True:
            # get next frame of the video
            success, input_frame = video_cap.read()
            if not success:
                break
            #input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            output_frame = input_frame.copy()
            img_shape = (video_height, video_width)
            img_center = (img_shape[0] / 2, img_shape[1] / 2)
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, input_frame)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, det_cat_id)
            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                output_frame,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names
            )

            if len(pose_results) > 0:
                # pose_results: list of pose_result
                # pose_result: bbox (left,top,right,bottom) and keypoints (array[kx3], x,y,score)
                midx = 0 # points to the bbox that locates at the center
                dist, min_dist = 0.0, 1000000000.0
                for idx, pose_result in enumerate(pose_results):
                    left, top, right, bottom = pose_result["bbox"][:-1]
                    bbox_center = ((top + bottom) / 2, (left + right) / 2)
                    dist = (img_center[0] - bbox_center[0]) ** 2 + (img_center[1] - bbox_center[1]) ** 2
                    if dist < min_dist:
                        midx, min_dist = idx, dist
                    
                pose_result = []
                pose_result.append(pose_results[midx])
                
                keypoints = pose_result[0]['keypoints'][:, :-1]
                assert keypoints.shape == (17, 2), f"Unexpected landmarks shape: {keypoints.shape}"

                pose_landmarks = np.array(keypoints, dtype=np.float32)
                landmarks = np.expand_dims(pose_landmarks, axis=0)
                landmarks = normalize_landmarks(landmarks)
                landmarks_tensor = torch.tensor(landmarks).float()
                output = model(landmarks_tensor)
                classes = torch.argmax(output, dim=1)
                class_int = classes.numpy()[0]
                prob_class = output[0][class_int].detach().numpy()
                if prob_class > 0.99:
                    action_counts[class_int] += 1
                # print(class_int)
                # print(classes)
        
        action_index = np.argmax(action_counts)
        most_action = index_label_dict[action_index]
        print(f"most_action: {most_action}, action_count: {action_counts}")
        

        action_type = most_action
        frame_idx = 0
        repetition_counter = RepetitionCounter(
            class_name=action_type,
            enter_threshold=0.9,
            exit_threshold=0.85
        )

        video_cap = cv2.VideoCapture(video_path)
        while True:
             # get next frame of the video
            success, input_frame = video_cap.read()
            if not success:
                break
            classify_prob = {v: 0 for k, v in index_label_dict.items()}
            #input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            output_frame = input_frame.copy()
            img_shape = (video_height, video_width)
            img_center = (img_shape[0] / 2, img_shape[1] / 2)
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, input_frame)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, det_cat_id)
            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                output_frame,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names
            )

            if len(pose_results) > 0:
                # pose_results: list of pose_result
                # pose_result: bbox (left,top,right,bottom) and keypoints (array[kx3], x,y,score)
                midx = 0 # points to the bbox that locates at the center
                dist, min_dist = 0.0, 1000000000.0
                for idx, pose_result in enumerate(pose_results):
                    left, top, right, bottom = pose_result["bbox"][:-1]
                    bbox_center = ((top + bottom) / 2, (left + right) / 2)
                    dist = (img_center[0] - bbox_center[0]) ** 2 + (img_center[1] - bbox_center[1]) ** 2
                    if dist < min_dist:
                        midx, min_dist = idx, dist
                    
                pose_result = []
                pose_result.append(pose_results[midx])
                
                keypoints = pose_result[0]['keypoints'][:, :-1]
                assert keypoints.shape == (17, 2), f"Unexpected landmarks shape: {keypoints.shape}"

                pose_landmarks = np.array(keypoints, dtype=np.float32)
                landmarks = np.expand_dims(pose_landmarks, axis=0)
                landmarks = normalize_landmarks(landmarks)
                landmarks_tensor = torch.tensor(landmarks).float()
                output = model(landmarks_tensor)
                classes = torch.argmax(output, dim=1)
                class_int = classes.numpy()[0]
                prob_class = output[0][class_int].detach().numpy()
                output_numpy = output.detach().numpy()[0]
                for action_index in range(num_classes):
                    action_n = index_label_dict[action_index]
                    classify_prob[action_n] = output_numpy[action_index]
                repetitions_count = repetition_counter(classify_prob)
            
            else:
                repetitions_count = repetition_counter.n_repeats
        
        
        mae = abs(gt_count - repetitions_count) / (gt_count + 1e-1)
        if abs(gt_count - repetitions_count) <= 1:
            obo = 1
        else:
            obo = 0
        testMAE.append(mae)
        testOBO.append(obo)
        print("gt count:", gt_count)
        print("repeat count:", repetitions_count)
    
    print(f"MAE: {np.mean(testMAE)}, OBO: {np.mean(testOBO)}")

