"""
数据集中一样的action type合并
"""
import os
import csv
import pandas as pd

if __name__ == '__main__':

    label_dir = '/root/autodl-tmp/data/RepCount/LLSP/annotation'
    cleaned_label_dir = '/root/autodl-tmp/data/RepCount/LLSP/cleaned/annotation'
    if not os.path.exists(cleaned_label_dir):
        os.makedirs(cleaned_label_dir)
    train_types = ['train', 'valid', 'test']

    for train_type in train_types:

        input_label_path = os.path.join(label_dir, train_type + '.csv')
        output_label_path = os.path.join(cleaned_label_dir, train_type + '.csv')

        output_list = []

        with open(input_label_path, 'r') as input_label_file:
            #label_pd = pd.read_csv(input_label_path)
            reader = csv.DictReader(input_label_file)
            for row in reader:
                video = dict(row).copy()
                if video['type'] == 'battle_rope':
                    video['type'] = 'battlerope'
                elif video['type'] == 'bench_pressing':
                    video['type'] = 'benchpressing'
                elif video['type'] == 'front_raise':
                    video['type'] = 'frontraise'
                elif video['type'] == 'jump_jack':
                    video['type'] = 'jumpjacks'
                elif video['type'] == 'pull_up':
                    video['type'] = 'pullups'
                elif video['type'] == 'push_up':
                    video['type'] = 'pushups'
                elif video['type'] == 'squant':
                    video['type'] = 'squat'
                output_list.append(video)
                    
        with open(output_label_path, 'w') as output_label_file:
            fieldnames = list(output_list[0].keys())
            csv_writer = csv.DictWriter(output_label_file, delimiter=',', fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(output_list)

