
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
from lib.models import load_model
from lib.utils import seed_everything
from lib.data.dataloader import load_data



def main():
    opt = Options().parse()


    '''SEED Everything'''
    seed_everything(SEED=opt.seed)


    data = load_data(opt)
    model = load_model(opt)
    best_model = train(opt,model,data)

    test(opt, data, best_model)

if __name__ == "__main__" :
    main()