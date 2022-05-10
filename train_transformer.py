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
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, encode_labels, PascalVOC_Dataset

from model.optim import ScheduledOptim

import model.constants as Constants
from model.transformer import Transformer
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

    # transformer.train()
    transformer.cuda()
    trained_samples = 0
    
    for batch_index, (feat, labels) in enumerate(train_loader):

        feat = feat.cuda()
        outputs = transformer(feat)

        # labels = labels.to(torch.int64)
        # labels = labels.cuda()
        labels = labels.argmax(dim=1)
        labels = labels.type('torch.LongTensor').cuda()

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
            optimizer.step()
            
    for name, param in transformer.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
        
    finish = time.time()
    
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    transformer.eval()

    test_loss = 0.0
    correct = 0.0

    for (feat, labels) in test_loader:

        feat = feat.cuda()
        
        outputs = transformer(feat)
        
        # labels = labels.to(torch.int64)
        # labels = labels.cuda()
        labels = labels.argmax(dim=1)
        labels = labels.type('torch.LongTensor').cuda()

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
        # test_loss / len(Y_test),
        # correct.float() / len(Y_test),
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
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    
    # d_key, d_value, d_model, d_inner, n_head, dropout = 256, 256, 2048, 2048, 8, 0.1
    # d_key, d_value, d_model, d_inner, n_head, dropout = 1025, 1025, 2050, 512, 2, 0.1
    if args.net == 'conv':
        d_key, d_value, d_model, d_inner, n_head, dropout = 64, 64, 256, 256, 4, 0.1
        # d_key, d_value, d_model, d_inner, n_head, dropout = 512, 512, 2048, 2048, 4, 0.1
        # d_key, d_value, d_model, d_inner, n_head, dropout = 257, 257, 514, 514, 2, 0.1

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

    ## CIFAR-100

    # cifar100_train_loader = get_cifar100_train_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=64,
    # )

    # cifar100_test_loader = get_cifar100_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=64,
    # )

    # print("[Train]  len(X):", len(X_train), "len(Y):", len(Y_train))
    # train_dataset = FeatureDataset(X_train, Y_train)
    
    # print("[Test]  len(X):", len(X_test), "len(Y):", len(Y_test))
    # test_dataset = FeatureDataset(X_test, Y_test)
    
    # train_loader = cifar100_train_loader
    # test_loader = cifar100_test_loader
    
    
    ## Pascal VOC
    
    data_dir = './data/'
    download_data=False
    # Imagnet values
    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    transformations = transforms.Compose([transforms.Resize((300, 300)),
#                                      transforms.RandomChoice([
#                                              transforms.CenterCrop(300),
#                                              transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
#                                              ]),                                      
                                      transforms.RandomChoice([
                                          transforms.ColorJitter(brightness=(0.80, 1.20)),
                                          transforms.RandomGrayscale(p = 0.25)
                                          ]),
                                      transforms.RandomHorizontalFlip(p = 0.25),
                                      transforms.RandomRotation(25),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean = mean, std = std),
                                      ])
        
    transformations_valid = transforms.Compose([transforms.Resize(330), 
                                          transforms.CenterCrop(300), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean = mean, std = std),
                                          ])
    
    dataset_train = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='train', 
                                      download=download_data, 
                                      transform=transformations, 
                                      target_transform=encode_labels)
    
    train_loader = DataLoader(dataset_train, batch_size=args.b, num_workers=4, shuffle=True)
    
    dataset_test = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=download_data, 
                                      transform=transformations_valid, 
                                      target_transform=encode_labels)
    
    test_loader = DataLoader(dataset_test, batch_size=args.b, num_workers=4)

    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09)
    # iter_per_epoch = len(X_train)
    iter_per_epoch = train_loader.__len__()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'convblock', settings.TIME_NOW
    ))

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