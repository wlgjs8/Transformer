import argparse
import sys
import os
import numpy as np

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print(os.getcwd())
sys.path.append('.')
sys.path.append('model')

from config import settings
from utils import get_network, get_cifar100_test_dataloader

OUTPUT_DIR = 'output'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=False, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-f', action='store_true', default=False, help='output features')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_cifar100_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    if args.f == True:
        net = torch.nn.Sequential(*list(net.children())[:-1])
    
        print(net)
        net.eval()

        outputs = []
        with torch.no_grad():
            for n_iter, (image, label) in enumerate(cifar100_test_loader):
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

                if args.gpu:
                    image = image.cuda()
                    label = label.cuda()
                    # print('GPU INFO.....')
                    # print(torch.cuda.memory_summary(), end='')
                
                output = net(image)
                outputs.append(output.reshape(-1, 2048).cpu())
        
        outputs = torch.cat([o for o in outputs], 0)
        print(outputs.shape)
        
        ## saving cifar100 features ([10000, 2048])
        # np.save( os.path.join(OUTPUT_DIR, 'cifar100_features.npy'), np.array(outputs))

    else:
        print(net)
        net.eval()

        correct_1 = 0.0
        correct_5 = 0.0
        total = 0

        with torch.no_grad():
            for n_iter, (image, label) in enumerate(cifar100_test_loader):
                print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

                if args.gpu:
                    image = image.cuda()
                    label = label.cuda()
                    print('GPU INFO.....')
                    print(torch.cuda.memory_summary(), end='')


                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()

        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')

        print()
        print("Top 1 correct: ", correct_1 / len(cifar100_test_loader.dataset))
        print("Top 5 correct: ", correct_5 / len(cifar100_test_loader.dataset))

        print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))