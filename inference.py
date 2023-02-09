import pandas as pd
import numpy as np
import os
import warnings
import cv2
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from sklearn.model_selection import train_test_split

import torch.onnx
import onnxruntime as onnxrt

from models.mlp_light import MLP

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

torch.multiprocessing.set_sharing_strategy('file_system')

def normalize_landmarks(all_landmarks):
    x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)

    y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
    y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)

    all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min)
    all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
    return all_landmarks

def show_image(img, figsize=(10, 10)):
    """show output PIL image"""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

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

if __name__ == "__main__":
    
    bbox_thr = 0.3
    kpt_thr = 0.3
    det_cat_id = 1
    radius = 4
    thickness = 1
    det_model, pose_model, dataset, dataset_info = config_detection()

    csv_label_path = 'action.csv'
    onnx_save_path = 'mlp_pose.onnx'
    input_video_dir = '/root/autodl-tmp/data/RepCount/LLSP/video/test'
    output_video_dir = '/root/autodl-tmp/data/RepCount/LLSP/video_visual_output'
    os.makedirs(output_video_dir, exist_ok=True)

    label_pd = pd.read_csv(csv_label_path)
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
    new_weights = torch.load(weight_path, map_location='cpu')
    model.layers.load_state_dict(new_weights)
    model.eval()
    
    for video_name in os.listdir(input_video_dir):
        video_path = os.path.join(input_video_dir, video_name)
        output_video_path = os.path.join(output_video_dir, video_name)
        print('video input path', video_path)
        print('video output path', output_video_path)
        video_cap = cv2.VideoCapture(video_path)

        # get some video parameters to generate output video with classification
        video_n_frame = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

        frame_idx = 0
        output_frame = None

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
                outputs=output_layer_names)
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

                output_frame = vis_pose_result(
                    pose_model,
                    input_frame,
                    pose_result,
                    dataset_info=dataset_info,
                    kpt_score_thr=kpt_thr,
                    radius=radius,
                    thickness=thickness,
                    show=False
                )
                
                pose_landmarks = np.array(keypoints, dtype=np.float32)
                landmarks = np.expand_dims(pose_landmarks, axis=0)
                landmarks = normalize_landmarks(landmarks)
                landmarks_tensor = torch.tensor(landmarks).float()
                output = model(landmarks_tensor)
                classes = torch.argmax(output, dim=1)
                class_int = classes.cpu().numpy()[0]
                prob_class = output[0][class_int].detach().cpu().numpy()
                # print(class_int)
                # print(classes)

                #save_picture = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
                #save_picture = output_frame
                frame_idx += 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (30, 30)
                fontScale = 0.7
                color = (0, 0, 255)
                thickness = 2
                show_text = 'class:' + index_label_dict[class_int] + '   prob:' + str(int(prob_class*100)/100.)
                output_frame = cv2.putText(output_frame, show_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                
            out_video.write(output_frame)
        
        video_cap.release()
        out_video.release()
        break
    
    input_names = [ "input" ]
    output_names = [ "output" ]

    dummy_input = torch.randn(1, 17*2)
    torch.onnx.export(model,
                     dummy_input,
                     onnx_save_path,
                     verbose=False,
                     input_names=input_names,
                     output_names=output_names,
                     export_params=True,
                     )

    onnx_session= onnxrt.InferenceSession(onnx_save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    onnx_inputs= {onnx_session.get_inputs()[0].name: np.zeros((1,17*2)).astype(np.float32)}
    onnx_output = onnx_session.run(None, onnx_inputs)
    img_label = onnx_output[0]
    print(img_label)
