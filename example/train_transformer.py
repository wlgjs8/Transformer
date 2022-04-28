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
from utils import get_network, CocoDataset, WarmUpLR, get_cifar100_test_dataloader, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from transformer import Transformer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    parser.add_argument('-weights', type=str, required=False, help='the weights file you want to test')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    # train_set = CocoDataset(set_name='train2017', split='TRAIN')
    test_set = CocoDataset(set_name='val2017', split='TEST')
    print(len(test_set))

    # train_loader = DataLoader(train_set, batch_size=1, collate_fn=train_set.collate_fn, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn, shuffle=False, num_workers=4, pin_memory=True)

    ## Pretrained ResNet50
    net.load_state_dict(torch.load(args.weights))
    net = torch.nn.Sequential(*list(net.children())[:-1])
    net.eval()

    cifar100_test_loader = get_cifar100_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    # outputs = []
    # with torch.no_grad():
    #     for n_iter, (image, label) in enumerate(cifar100_test_loader):
    #         print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

    #         if args.gpu:
    #             image = image.cuda()
    #             label = label.cuda()
    #             # print('GPU INFO.....')
    #             # print(torch.cuda.memory_summary(), end='')
            
    #         output = net(image)
    #         outputs.append(output.cpu())

    # print(np.array(outputs).shape)
    # print(outputs[0].shape)

    temp = np.arange(128*2048)
    temp = temp.reshape((128,2048, 1, 1))
    temp = torch.Tensor(temp)

    d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 512, 2048, 8, 0.1
    print(d_key, d_value, d_model, d_inner, n_head, dropout)

    transformer = Transformer(
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        dropout=dropout
    )
    print(transformer(temp))