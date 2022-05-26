import os
import cv2
import math
import torch
import random

import numpy as np
import torch.nn as nn

from PIL import ImageFilter
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support as score


def image_entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])


class add_smartfilter(object):
    def __init__(self):
       self.x=3
    def __call__(self, image):
        filtered =[]
        IDX=[]
        sz=image.size
        ENT=[]
        for i in range(1,int(sz[0]/4)):
           image_filtered = image.filter(ImageFilter.GaussianBlur(radius=i))                        
           ENT.append(image_entropy(image_filtered))   
        idx=ENT.index(max(ENT))  
        IDX.append(idx)
        image_filtered = image.filter(ImageFilter.GaussianBlur(radius=idx)) 
        return image_filtered
        
'''SEED Everything'''
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # keep True if all the input have same size.

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
