
import os
import time
import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import models

from test import test
from train import train 
from options import Options
from lib.models.GLP import GLP
from lib.utils import seed_everything
from lib.data.dataloader import load_data



def main():
    opt = Options().parse()


    '''SEED Everything'''
    seed_everything(SEED=opt.seed)


    data = load_data(opt)
    model = train(opt,data)

    test(opt, data, model)

if __name__ == "__main__" :
    main()