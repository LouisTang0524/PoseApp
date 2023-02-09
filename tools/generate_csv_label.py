"""
根据提取出的视频目录包含的视频类型生成action.csv
action.csv: action -> label
"""
import pandas as pd
import os

csv_name = 'action.csv'
#train_folder = '/root/autodl-tmp/data/RepCount/LLSP/extracted/train'
train_folder = '/root/autodl-tmp/data/RepCount/LLSP/cleaned/extracted/train'
videos = os.listdir(train_folder)

label = {}
label['action'] = []
label['label'] = []
for cnt, action_type in enumerate(videos):
    label['action'].append(action_type)
    label['label'].append(cnt)
label_pd = pd.DataFrame.from_dict(label)
label_pd.to_csv(csv_name)