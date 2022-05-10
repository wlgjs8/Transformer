import os
import sys
import re
import datetime

import numpy as np
from PIL import Image

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms

# from torchvision.transforms import transform_COCO
from torch.utils.data import DataLoader, Dataset
from config import settings

from resnet import resnet18, resnet34, resnet50, resnet101, conv_blocks

from pycocotools.coco import COCO

import torchvision.datasets.voc as voc

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def get_network(args):

    if args.net == 'resnet50':
        net = resnet50()

    elif args.net == 'resnet101':
        net = resnet101()

    elif args.net == 'resnet18':
        net = resnet18()

    elif args.net == 'resnet34':
        net = resnet34()
        
    elif args.net == 'conv':
        net = conv_blocks()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net


def get_cifar100_train_dataloader(mean, std, batch_size=64, num_workers=4, shuffle=True):
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_cifar100_test_dataloader(mean, std, batch_size=64, num_workers=4, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


### Pascal VOC
def get_PascalVOC2012_train_dataloader(batch_size=64, num_workers=4, shuffle=True):
    
    transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomChoice([
                                        transforms.ColorJitter(brightness=(0.80, 1.20)),
                                        transforms.RandomGrayscale(p = 0.25)
                                        ]),
                                    transforms.RandomHorizontalFlip(p = 0.25),
                                    transforms.RandomRotation(25),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(
                                        mean = settings.PASCAL_TRAIN_MEAN, 
                                        std = settings.PASCAL_TRAIN_STD),
                                    ])

    pascal_training = PascalVOC_Dataset(
        root='./data', 
        year='2012', 
        image_set='train', 
        download=False, 
        transform=transform_train,
        target_transform=encode_labels
    )

    pascal_training_loader = DataLoader(
        pascal_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return pascal_training_loader


def get_PascalVOC2012_test_dataloader(batch_size=64, num_workers=4, shuffle=True):

    transform_test = transforms.Compose([transforms.Resize((224, 224)), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(
                                            mean = settings.PASCAL_TRAIN_MEAN, 
                                            std = settings.PASCAL_TRAIN_STD),
                                    ])
    
    pascal_test = PascalVOC_Dataset(
        root='./data', 
        year='2012', 
        image_set='val', 
        download=False, 
        transform=transform_test,
        target_transform=encode_labels
    )

    pascal_test_loader = DataLoader(
        pascal_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return pascal_test_loader


class CocoDataset(Dataset):
    def __init__(self, root_dir='/home/jeeheon/Documents/Transformer', set_name='val2017', split='TRAIN', img_size=512, transform=None):

        super().__init__()
        self.root_dir = os.getcwd()
        # print('COCO ROOT DIR : ', os.getcwd())
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'coco/annotations', 'instances_' + self.set_name + '.json'))

        whole_image_ids = self.coco.getImgIds()

        self.image_ids = []
        self.no_anno_list = []

        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0:
                self.no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)
            
        # self.load_classes()
        self.split = split
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, idx):

        visualize = True

        image, (w, h) = self.load_image(idx)

        annotation = self.load_annotations(idx)

        boxes = torch.FloatTensor(annotation[:, :4])
        labels = torch.LongTensor(annotation[:, 4])

        if labels.nelement() == 0:
            visualize = True

        if self.transform is not None:
            # image, boxes, labels, segmentations = self.transform(image, boxes, labels, self.split)
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels

    def __len__(self):  
        return len(self.image_ids)

    def load_image(self, idx):
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path = os.path.join(self.root_dir, 'coco', self.set_name, image_info['file_name'])
        image = Image.open(path).convert('RGB')
        #Image.Resampling.BILINEAR = 2
        image = image.resize((self.img_size, self.img_size), resample=2)

        trans = transforms.ToTensor()
        image = trans(image)
        return image, (image_info['width'], image_info['height'])

    def load_annotations(self, idx):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for index, a in enumerate(coco_annotations):

            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id']
            annotations = np.append(annotations, annotation, axis=0)
        
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    
    ls = target['annotation']['object']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
  
    return torch.from_numpy(k)


class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)
        
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)