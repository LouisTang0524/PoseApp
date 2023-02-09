"""
MLP的定义
"""

import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from sklearn.model_selection import train_test_split


class MLP(pl.LightningModule):

    def __init__(self, X, y, learning_rate, seed, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.layers = nn.Sequential(
            nn.Linear(17*2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.seed = seed
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average="macro")

    def forward(self, x):
        x = self.layers(x)
        soft_output = x.softmax(dim=-1)
        return soft_output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.loss(y_hat, y)

        y_pred = y_hat.softmax(dim=-1)
        y_tgt = y
        acc = self.train_acc(y_pred, y_tgt)
        f1 = self.train_f1(y_pred, y_tgt)

        self.train_auroc.update(y_pred, y_tgt)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        self.log("train_f1", f1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.loss(y_hat, y)
        
        pred = y_hat.softmax(dim=-1)
        self.val_acc.update(pred, y)
        self.val_f1.update(pred, y)
        self.val_auroc.update(pred, y)
        return loss
    
    def validation_epoch_end(self, val_step_outputs):
        loss = sum(val_step_outputs) / len(val_step_outputs)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.75,
            patience=6,
            verbose=1,
            mode='min',
            cooldown=0,
            min_lr=10e-7
        )
        optimizer_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer_dict
    
    def setup(self, stage):
        X = self.X
        y = self.y

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.85, random_state=self.seed)
        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        
        self.X_train_scaled = X_train
        self.X_val_scaled = X_val

        self.y_train_scaled = y_train
        self.y_val_scaled = y_val

    def train_dataloader(self):
        dataset = TensorDataset(torch.FloatTensor(self.X_train_scaled), torch.LongTensor(self.y_train_scaled))
        train_loader = DataLoader(dataset, batch_sampler=16, num_workers=8, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(torch.FloatTensor(self.X_val_scaled), torch.LongTensor(self.y_val_scaled))
        val_loader = DataLoader(val_dataset, batch_sampler=16, num_workers=8, shuffle=False)
        return val_loader
    
    def training_epoch_end(self, trainning_step_outputs):
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        train_auroc = self.train_auroc.compute()

        self.log("epoch_train_accuracy", train_accuracy)
        self.log("epoch_train_f1", train_f1)

        self.train_acc.reset()
        self.train_f1.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, " \
                f"f1: {train_f1:.4}, auroc: {train_auroc:.4}")
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.tensor(validation_step_outputs).mean()
        val_accuracy = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()

        self.log("val_accuracy", val_accuracy)
        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)
        self.log("val_auroc", val_auroc)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        print(f"validation accuracy: {val_accuracy:.4} " \
                f"f1: {val_f1:.4}, auroc: {val_auroc:.4}")
