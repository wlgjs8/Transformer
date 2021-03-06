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
from utils import get_network, WarmUpLR, get_PascalVOC2012_train_dataloader, get_PascalVOC2012_test_dataloader, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, encode_labels

from model.optim import ScheduledOptim

import model.constants as Constants
from model.transformer import Transformer
from model.optim import ScheduledOptim


def train(epoch):

    start = time.time()

    transformer.train()
    transformer.cuda()
    trained_samples = 0
    
    for batch_index, (feat, labels) in enumerate(train_loader):

        feat = feat.cuda()
        outputs = transformer(feat)

        # labels = labels.to(torch.int64)
        # labels = labels.cuda()
        ### PascalVOC ###
        labels = labels.argmax(dim=1)
        labels = labels.type('torch.LongTensor').cuda()

        # print('train outputs : ', outputs.shape)
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
            total_samples=len(train_loader.dataset)
        ))

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()
            
    # for name, param in transformer.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        
    finish = time.time()
    
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    transformer.eval()

    test_loss = 0.0
    correct = 0.0
    correct_1 = 0.0
    correct_5 = 0.0

    for (feat, labels) in test_loader:

        feat = feat.cuda()
        outputs = transformer(feat)
        
        # labels = labels.to(torch.int64)
        # labels = labels.cuda()
        ### PascalVOC ###
        labels = labels.argmax(dim=1)
        labels = labels.type('torch.LongTensor').cuda()

        # print(outputs)
        # print(labels)

        loss = loss_function(outputs, labels)
        
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        label = labels.view(labels.size(0), -1).expand_as(pred)
        corrects = pred.eq(label).float()
        correct_5 += corrects[:, :5].sum()
        correct_1 += corrects[:, :1].sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        # test_loss / len(Y_test),
        # correct.float() / len(Y_test),
        test_loss / test_loader.__len__(),
        correct.float() / test_loader.__len__(),
        finish - start)
    )
    print("Top 1 correct: ", correct_1 / len(test_loader.dataset))
    print("Top 5 correct: ", correct_5 / len(test_loader.dataset))
    print()

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / test_loader.__len__(), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / test_loader.__len__(), epoch)

    return correct.float() / test_loader.__len__()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')    
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    
    if args.net == 'resnet34':
        # d_key, d_value, d_model, d_inner, n_head, dropout = 256, 256, 2048, 2048, 8, 0.1
        d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 768, 2048, 8, 0.1
    else:
        d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 256, 256, 4, 0.1

    print(d_key, d_value, d_model, d_inner, n_head, dropout)

    net = get_network(args)

    transformer = Transformer(
        net=net,
        n_head=n_head,
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        d_inner=d_inner,
        dropout=dropout,
    )
    
    train_loader = get_PascalVOC2012_train_dataloader(
        num_workers=4,
        batch_size=args.b
    )
    test_loader = get_PascalVOC2012_test_dataloader(
        num_workers=4,
        batch_size=args.b
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = train_loader.__len__()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'resnet34_transformer', settings.TIME_NOW
    ))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{transformer}-{epoch}-{type}.pth')
    
    best_acc = 0.0
    
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
        
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(transformer='resnet34', epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            best_acc = acc
            continue
        
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(transformer='resnet34', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(transformer.state_dict(), weights_path)
            
    writer.close()