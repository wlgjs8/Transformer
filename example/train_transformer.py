import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
sys.path.append('model')

from config import settings
from utils import get_network, CocoDataset, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    train_set = CocoDataset(set_name='train2017', split='TRAIN')
    test_set = CocoDataset(set_name='val2017', split='TEST')
    print(len(train_set))


    train_loader = DataLoader(train_set, batch_size=1, collate_fn=train_set.collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=train_set.collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_set, batch_size=4, collate_fn=None, shuffle=False, num_workers=4, pin_memory=True)
    
    # for i, (images, boxes, labels) in enumerate(train_loader):
    #     print(boxes)
    #     print()

    #     images = images.cuda()
    #     boxes = [b.cuda() for b in boxes]
    #     labels = [l.cuda() for l in labels]

    # return 0 