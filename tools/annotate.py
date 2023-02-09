"""
抽取视频的关键帧并按照action type分目录组织
"""
import csv
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

video_root = "/root/autodl-tmp/data/RepCount/LLSP/video"
label_dir = "/root/autodl-tmp/data/RepCount/LLSP/cleaned"
save_root = "/root/autodl-tmp/data/RepCount/LLSP/cleaned/extracted"

video_types = ['train', 'valid', 'test']

for video_type in video_types:
    print(f"Processing {video_type}...")
    label_filename = video_type + '.csv'
    save_dir = os.path.join(save_root, video_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    video_dir = os.path.join(video_root, video_type)
    label_path = os.path.join(label_dir, label_filename)
    df = pd.read_csv(label_path)

    # 从csv中提取label
    file2label = {}
    num_idx = 4
    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        action_type = df.loc[i, 'type']
        label_tmp = df.values[i][num_idx:].astype(np.float64)
        label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
        file2label[filename] = [label_tmp, action_type]
    
    # 对每一个视频
    for video in tqdm(file2label):
        video_path = os.path.join(video_dir, video)
        # 抽取视频的每一帧保存至frames中
        cap = cv2.VideoCapture(video_path)
        frames = []
        if cap.isOpened():
            while True:
                success, frame_bgr = cap.read()
                if success is False:
                    break
                frames.append(frame_bgr)
        cap.release()

        count = 0
        label, action_type = file2label[video]
        for frame_idx in label:
            if frame_idx >= len(frames):
                continue
            frame_ = frames[frame_idx]
            video_save_dir = os.path.join(save_dir, action_type, video)
            if not os.path.isdir(video_save_dir):
                os.makedirs(video_save_dir)
            save_path = os.path.join(video_save_dir, str(count)+'.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1