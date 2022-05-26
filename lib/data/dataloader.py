"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torchvision
import torch
from torchvision import transforms
from lib.utils import add_smartfilter
from torch.utils.data import DataLoader
from lib.data.datasets import load_Caltech101, load_Navon
from torchvision.datasets import Caltech101, ImageFolder


class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)
    
  
    ## Caltech101
    if opt.dataset in ['caltech101']:

        if opt.add_smartfilter:
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(opt.icrop),
                transforms.RandomHorizontalFlip(),
                add_smartfilter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

            val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(opt.isize),
                transforms.CenterCrop(opt.icrop),
                add_smartfilter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        else: 
                train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(opt.icrop),
                    # transforms.Resize(opt.isize),
                    # transforms.CenterCrop(opt.icrop),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

                val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(opt.isize),
                    transforms.CenterCrop(opt.icrop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])


      
        train_ds, valid_ds, test_ds = load_Caltech101(opt.dataroot,train_transform,val_transform)
        

    elif opt.dataset in ['navon']:
        if opt.add_smartfilter:
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(opt.icrop),
                transforms.RandomHorizontalFlip(),
                add_smartfilter(),
                transforms.ToTensor(),
                transforms.Normalize((0),(1))
        ])

            val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(opt.icrop),
                add_smartfilter(),
                transforms.ToTensor(),
                transforms.Normalize((0),(1))
        ])
        else: 
                train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(opt.icrop),
                    # transforms.RandomResizedCrop(opt.icrop),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0],std=[1])
        ])

                val_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(opt.icrop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0],std=[1])
        ])
        train_ds, valid_ds, test_ds = load_Navon(opt.dataroot,train_transform,val_transform)

    # FOLDER
    else:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
        valid_ds = ImageFolder(os.path.join(opt.dataroot, 'val'), transform)
        test_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    

    if opt.dataset in ['navon']:
        train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True,
                          drop_last=True,num_workers=opt.workers)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, 
                          drop_last=False,num_workers=opt.workers)
        test_dl_local = DataLoader(dataset=test_ds[0], batch_size=opt.batchsize, shuffle=False,
                              drop_last=False,num_workers=opt.workers)

        test_dl_global = DataLoader(dataset=test_ds[1], batch_size=opt.batchsize, shuffle=False,
                              drop_last=False,num_workers=opt.workers)
        
        test_dl = [test_dl_local, test_dl_global]

    else: 
        train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True,
                          drop_last=True,num_workers=opt.workers)
        valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, 
                          drop_last=False,num_workers=opt.workers)
        test_dl = valid_dl

    return Data(train_dl, valid_dl, test_dl)
