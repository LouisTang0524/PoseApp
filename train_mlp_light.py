import pandas as pd
import numpy as np
import os
import csv
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

from models.mlp_light import MLP

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

def obtain_landmark_label(csv_path, all_landmarks, all_labels, label2index):
    file_separator = ','
    n_landmarks = 17
    n_dimensions = 2
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
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
    train_landmarks, train_labels = obtain_landmark_label(train_csv, train_landmarks, train_labels, action2index)

    train_landmarks = np.array(train_landmarks)
    train_labels = np.array(train_labels)
    train_landmarks = normalize_landmarks(train_landmarks)

    return train_landmarks, train_labels

if __name__ == "__main__":

    csv_label_path = 'action.csv'
    root_dir = '/root/autodl-tmp/data/RepCount/LLSP/cleaned'
    data_root = os.path.join(root_dir, 'extracted')

    train_csv = os.path.join(root_dir, 'train.csv')
    valid_csv = os.path.join(root_dir, 'valid.csv')
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

    train_landmarks, train_labels = csv2data(train_csv, action2index)
    valid_landmarks, valid_labels = csv2data(valid_csv, action2index)
    test_landmarks, test_labels = csv2data(test_csv, action2index)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )
    ckpt_callback = ModelCheckpoint(
        mode='min',
        monitor='val_loss',
        dirpath='./saved_weights',
        filename='{epoch}-{val_loss:.2f}',
        every_n_epochs=1
    )

    model = MLP(train_landmarks, train_labels, valid_landmarks, valid_labels, learning_rate=1e-3, seed=42, num_classes=num_classes)
    trainer = pl.Trainer(auto_lr_find=True)
    trainer.tune(model)
    print("Learning rate:", model.learning_rate)
    trainer = pl.Trainer(callbacks=[early_stop_callback, ckpt_callback], accelerator='gpu', devices=1)
    trainer.fit(model)

    print(f"best loss: {ckpt_callback.best_model_score.item():.5g}")

    weights = model.layers.state_dict()
    torch.save(weights, 'best_weights.pth')
