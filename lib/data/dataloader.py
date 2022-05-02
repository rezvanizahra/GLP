"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os

from torchvision import transforms
from lib.utils import add_gussianblur
from torch.utils.data import DataLoader
from lib.data.datasets import load_Caltech101, CocoClsDataset
from torchvision.datasets import Caltech101, ImageFolder
from torchvision.datasets.coco import CocoDetection


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
    
    if opt.add_gussianblur:
            train_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomResizedCrop(opt.icrop),
                transforms.RandomHorizontalFlip(),
                add_gussianblur(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

            val_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(opt.isize),
                transforms.CenterCrop(opt.icrop),
                add_gussianblur(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    else: 
            train_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomResizedCrop(opt.icrop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

            val_transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(opt.isize),
                transforms.CenterCrop(opt.icrop),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])


    ## Caltech101
    if opt.dataset in ['caltech101']:
      
        train_ds, valid_ds, test_ds = load_Caltech101(opt.dataroot,train_transform,val_transform)
        
    ## coco
    elif opt.dataset in ['coco']:
        print('coco')
        # train_ds =CocoDetection(root=os.path.join(opt.dataroot, 'train2017'), 
        #                            annFile=os.path.join(opt.dataroot, 'annotations/instances_train2017.json'),
        #                            # img_dir='train2017',
        #                            transform = train_transform)
        # valid_ds =CocoDetection(root=os.path.join(opt.dataroot, 'valid2017'), 
        #                            annFile=os.path.join(opt.dataroot, 'annotations/instances_val2017.json'),
        #                            # img_dir='train2017',
        #                            transform = val_transform)
        train_ds = CocoClsDataset(root_dir=opt.dataroot, 
                                   ann_file='annotations/instances_train2017.json',
                                   img_dir='train2017',
                                   # bg_bboxes_file=None,#'bg_bboxes/coco_train_bg_bboxes.log',
                                   phase='train', 
                                   transform = train_transform)
        valid_ds = CocoClsDataset(root_dir=opt.dataroot, 
                                     ann_file='annotations/instances_val2017.json',
                                     img_dir='val2017',
                                     # bg_bboxes_file=None,#'bg_bboxes/coco_val_bg_bboxes.log',
                                     phase='test', 
                                     transform = val_transform)


        test_ds = None

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
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True,
                          drop_last=True,num_workers=opt.workers)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, 
                          drop_last=False,num_workers=opt.workers)
    if not opt.dataset in ['coco']:
        test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False,
                              drop_last=False,num_workers=opt.workers)
    else: 
        test_dl = valid_dl

    return Data(train_dl, valid_dl, test_dl)
