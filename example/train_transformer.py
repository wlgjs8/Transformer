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

from train_test import train, eval_training
    
from model.optim import ScheduledOptim

import model.constants as Constants
from model.transformer import Transformer
from model.optim import ScheduledOptim


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    
    d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 2048, 2048, 8, 0.1
    print(d_key, d_value, d_model, d_inner, n_head, dropout)

    transformer = Transformer(
        n_head=n_head,
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        d_inner=d_inner,
        dropout=dropout,
    )
    
    X_train = torch.Tensor(np.load('output/cifar100_train_features.npy'))
    Y_train = torch.Tensor(np.load('output/cifar100_train_labels.npy'))
    print("[Train]  len(X):", len(X_train), "len(Y):", len(Y_train))
    
    X_test = torch.Tensor(np.load('output/cifar100_test_features.npy'))
    Y_test = torch.Tensor(np.load('output/cifar100_test_labels.npy'))
    print("[Test]  len(X):", len(X_test), "len(Y):", len(Y_test))
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09)
    iter_per_epoch = len(X_train)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    
    ## log 작성??
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'transformer', settings.TIME_NOW
    ))
    input_tensor = torch.Tensor(1, 3, 32, 32)           ## input tensor 사이즈 변경 필요
    # input_tensor = np.arange(1, 2048, 1)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(transformer, input_tensor)
    ##
    

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{transformer}-{epoch}-{type}.pth')
    
    best_acc = 0.0
    
    for epoch in range(1, settings.EPOCH + 1):
      
        train(epoch)
        acc = eval_training(epoch)
        
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(transformer='transformer', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            best_acc = acc
            continue
        
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(transformer='transformer', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            
    writer.close()