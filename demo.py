"""
A demo that draws the confidences
while detecting the pose of each frame
and counts the moves at the same time.
"""
import pandas as pd
import numpy as np
import os
import warnings
import cv2
import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch

from models.mlp_light import MLP
from tools.counter import RepetitionCounter
from tools.visualizer import PoseClassificationVisulizer


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

    action_label_path = 'action.csv'
    action_type = 'pushups'
    label_pd = pd.read_csv(action_label_path)
    index_label_dict = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index_label_dict[label] = action
    
    num_classes = len(index_label_dict)

    model = MLP(None, None, 1e-3, seed=42, num_classes=num_classes)
    weight_path = 'best_weights.pth'
    new_weights = torch.load(weight_path, map_location='cuda:0')
    model.layers.load_state_dict(new_weights)
    model.eval()

    bbox_thr = 0.3
    kpt_thr = 0.3
    det_cat_id = 1
    radius = 4
    thickness = 1
    det_model, pose_model, dataset, dataset_info = config_detection()

    # Counter
    repetition_counter = RepetitionCounter(
        class_name=action_type,
        enter_threshold=0.95,
        exit_threshold=0.8
    )

    input_video_path ='/root/autodl-tmp/data/RepCount/LLSP/video/test/stu4_46.mp4'
    output_video_path = 'demo_outputs/stu4_46.mp4'
    print("Input video path:", input_video_path)
    
    video_cap = cv2.VideoCapture(input_video_path)
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

    # Renderer
    pose_classification_visualizer = PoseClassificationVisulizer(
        class_name=action_type,
        plot_x_max=video_n_frames,
        plot_y_max=10
    )

    action_prob = []

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
        

        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=classify_prob,
            pose_classification_filtered=classify_prob,
            repetitions_count=repetitions_count
        )

        action_prob.append(classify_prob[action_type])

        out_video.write(np.array(output_frame))
    out_video.release()
    print("Output video path", output_video_path)

    x = range(len(action_prob))
    y = action_prob
    plt.figure(figsize=(40, 10))
    plt.plot(x, y, 'b--*')
    plt.xlabel("frames")
    plt.ylabel("prob")
    plt.savefig("demo_stu4_46.jpg")
    print("Graph: demo_stu4_46.jpg")
