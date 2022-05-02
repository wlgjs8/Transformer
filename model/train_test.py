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


def train(transformer, epoch):
    
    start = time.time()
    transformer.train()
    for batch_index, (feat, labels) in enumerate(zip(X_train, Y_train)):
    
        if args.gpu:
            labels = labels.cuda()
            feat = feat.cuda()

        optimizer.zero_grad()
        outputs = transformer(feat)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(Y_train) + batch_index + 1

        # last_layer = list(transformer.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer._optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(feat),
            total_samples=len(Y_train)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
        #     optimizer.step()
            
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

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (feat, labels) in zip(X_test, Y_test):

        if args.gpu:
            feat = feat.cuda()
            labels = labels.cuda()

        outputs = transformer(feat)
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
        test_loss / len(Y_test),
        correct.float() / len(Y_test)),
        finish - start
    )
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(Y_test), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(Y_test), epoch)

    return correct.float() / len(Y_test)
