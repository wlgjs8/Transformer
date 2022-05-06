import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
sys.path.append('model')

from config import settings
from utils import get_network, CocoDataset, WarmUpLR, get_cifar100_train_dataloader, get_cifar100_test_dataloader, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from model.optim import ScheduledOptim

import model.constants as Constants
from model.resnet import RegressionModel
from model.optim import ScheduledOptim


class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


def train(epoch):

    start = time.time()

    regression_model.cuda()
    trained_samples = 0
    
    for batch_index, (feat, labels) in enumerate(train_loader):

        feat = feat.cuda()
        outputs = regression_model(feat)

        labels = labels.to(torch.int64)
        labels = labels.cuda()

        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * train_loader.__len__() + batch_index + 1
        trained_samples += len(feat)

        print('Training Epoch: {epoch} [{trained}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained=trained_samples,
            total_samples=50000
        ))

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            optimizer.step()
            
    for name, param in regression_model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        
    finish = time.time()
    
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    regression_model.eval()

    test_loss = 0.0
    correct = 0.0

    for (feat, labels) in test_loader:

        feat = feat.cuda()
        
        outputs = regression_model(feat)
        
        labels = labels.to(torch.int64)
        labels = labels.cuda()

        loss = loss_function(outputs, labels)
        
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / test_loader.__len__(),
        correct.float() / test_loader.__len__(),
        finish - start)
    )
    print()

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / test_loader.__len__(), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / test_loader.__len__(), epoch)

    return correct.float() / test_loader.__len__()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')    
    args = parser.parse_args()
    
    if args.net == 'resnet34':
        d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 256, 256, 4, 0.1

    print(d_key, d_value, d_model, d_inner, n_head, dropout)

    net = get_network(args)

    cifar100_train_loader = get_cifar100_train_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=64,  
    )

    cifar100_test_loader = get_cifar100_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=64,
    )

    regression_model = RegressionModel(
        net=net,
    )
    
    train_loader = cifar100_train_loader
    test_loader = cifar100_test_loader

    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    optimizer = optim.Adam(regression_model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    iter_per_epoch = train_loader.__len__()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'regression', settings.TIME_NOW
    ))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{transformer}-{epoch}-{type}.pth')
    
    best_acc = 0.0
    
    for epoch in range(1, settings.EPOCH + 1):
        train(epoch)
        acc = eval_training(epoch)
        
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(transformer='regression', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            best_acc = acc
            continue
        
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(transformer='regression', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            
    writer.close()