import os
import csv
import json
import time
import torch
import argparse

import numpy as np 
import torch.nn as nn
import torchvision.datasets as datasets

from PIL import Image
from torchvision import models
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    precision, recall, f1score, support = score(target.cpu(), pred[:1][0].cpu(),\
                                                    average = 'weighted', zero_division=1)

    res.append(f1score*100.0)
    res.append(precision*100.0)
    res.append(recall*100.0)
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fscore = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        # print(data)
        input = data[0].cuda()
        target = data[1].squeeze(-1).long().cuda()
        val_loader_len = len(val_loader.dataset)

        # compute output
        with torch.no_grad():
            output = model(input)

        # measure accuracy and record loss
        prec1, prec5, f1score, precision, recall = accuracy(output.data, target, topk=(1, 5))

        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        fscore.update(f1score, input.size(0))
        precisions.update(precision, input.size(0))
        recalls.update(recall, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Top@1 {top1.avg:.3f} Top@5 {top5.avg:.3f} F1-Score {fscore.avg:.3f} \
            precision {precisions.avg:.3f} recall {recalls.avg:.3f} '
        .format(top1=top1, top5=top5, fscore=fscore, precisions=precisions , recalls=recalls))


def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
  
    args = parser.parse_args()
    return args


def main():

    args = parse()
    val_transforms= transforms.Compose([transforms.Resize(150),
   									 transforms.CenterCrop(128),              
                                    transforms.ToTensor(),                     
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])])
    data_test = datasets.ImageNet(root= args.data[1], split = 'val',
                       transform=val_transforms)

    test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        model = model.cuda()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model = model.cuda()

    validate(test_loader, model)

    


if __name__ == '__main__':
    main()