#!/usr/bin/env python3

import os
import time
import pickle
import argparse
from tkinter import Label
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch
from typing import Dict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from detector.bbox_regressor import ObjectDetector
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from detector.coco_detection_dataset import CocoDataset, get_transform, collate_fn


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PIN_MEM = True if DEVICE == 'cuda' else False
# Loss weights
LABELS = 1.0
BBOX = 1.0


def main(cmd_args: Dict):
    coco_data_train = CocoDataset(
        root=cmd_args['coco_imgs'],
        annFile=cmd_args['coco_anns_train'],
        transforms=get_transform()
    )
    coco_data_val = CocoDataset(
        root=cmd_args['coco_imgs'],
        annFile=cmd_args['coco_anns_val'],
        transforms=get_transform()
    )

    coco_dataloader_train = DataLoader(
        dataset=coco_data_train,
        batch_size=cmd_args['batch_size'],
        shuffle=cmd_args['shuffle_data'],
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=PIN_MEM
    )
    coco_dataloader_val = DataLoader(
        dataset=coco_data_val,
        batch_size=cmd_args['batch_size'],
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=PIN_MEM
    )

    labels_train = np.array(coco_dataloader_train.dataset.coco.getCatIds())
    labels_val = np.array(coco_dataloader_val.dataset.coco.getCatIds())
    le_train = LabelEncoder()
    le_val = LabelEncoder()
    le_train.fit_transform(labels_train)
    le_val.fit_transform(labels_val)

    train_epoch_len = len(coco_dataloader_train)
    val_epoch_len = len(coco_dataloader_val)

    print(f'[INFO] Total training steps: {train_epoch_len}')
    print(f'[INFO] Total validation steps: {val_epoch_len}')

    resnet = resnet50(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    objDetect = ObjectDetector(resnet, len(le_train.classes_))
    objDetect.to(DEVICE)

    classLossFunc = CrossEntropyLoss()
    bboxLossFunc = MSELoss()

    opt = Adam(objDetect.parameters(), lr=cmd_args['learning_rate'])

    # initialize a dictionary to store training history
    H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(cmd_args['num_epochs'])):
        # set the model in training mode
        objDetect.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # loop over the training set
        for idx, (images, labels, bboxes) in enumerate(tqdm(coco_dataloader_train)):
            # send the input to the device
            images = list(image.to(DEVICE) for image in images)
            bboxes = list(bbox.to(DEVICE) for bbox in bboxes)
            labels = list(label.to(DEVICE) for label in labels)
            
            # perform a forward pass and calculate the training loss
            predictions = objDetect(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            totalLoss.backward()
            opt.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += totalLoss
            trainCorrect += (predictions[1].argmax(1) == labels).type(
                torch.float).sum().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco_imgs', type=Path, help='Path to the MS COCO images directory')
    parser.add_argument('coco_anns_train', type=Path, help='Path to the MS COCO formated training annotations file')
    parser.add_argument('coco_anns_val', type=Path, help='Path to the MS COCO formatted validation annotations file')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='The training batch size')
    parser.add_argument('-s', '--shuffle-data', action='store_const', const=True, default=False, help='Shuffle the training data for each epoch')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='The learning rate of the optimizer')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='The number of epoch for training')
    args = vars(parser.parse_args())

    main(args)