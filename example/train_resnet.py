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

from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(mnist_training_loader):

        if arge.gpu:
            labels = labels.cuda()
            images = iamges.cuda()

        optimizers.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(mnist_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar()


@torch.no_grad()
def eval_training(epoch=0, tb=True):


    return correct.float() / len( __ test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    

    net = get_network(args)

    #data preprocessing:
    mnist_training_loader = get_trai