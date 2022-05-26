import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models as torch_models

from options import Options
from lib.utils import seed_everything
from lib.models.models import GAS, GLP
from lib.models.inception_v3 import inception_v3




def load_glp(opt):
    """ Load glp model based on the model name.
    Arguments:
        opt {[argparse.Namespace]} -- options
       
    Returns:
        [model] -- Returned model
    """
    
    if opt.model == 'GA-Resnet': 
        #load pretrain imagenet model
        print("=> using pre-trained model '{}'".format(opt.model))
        pre_model = torch_models.resnet18(pretrained=True)
        pre_model= nn.Sequential(*list(pre_model.children())[:-2])

    elif opt.model == 'GA-Inception': 
        #load pretrain imagenet model
        print("=> using pre-trained model '{}'".format(opt.model))
        pre_model = inception_v3(pretrained=True)
        num_ftrs = pre_model.AuxLogits.fc.in_features
        pre_model.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        nn.init.xavier_uniform_(pre_model.AuxLogits.fc.weight)

    #load gas model
    gas_model = GAS(opt)
    gas_model.load_state_dict(torch.load(opt.gas_path))
    #freezing all layers: 
    for param in gas_model.parameters():
        param.requires_grad = False
    gas_model = nn.Sequential(*list(gas_model.children())[:-2])
    model = GLP(opt, pre_model,gas_model)
    nn.init.xavier_uniform_(model.fc.weight)
    return model


def load_model(opt):
    """ Load a model based on the model name.
    Arguments:
        opt {[argparse.Namespace]} -- options
       
    Returns:
        [model] -- Returned model
    """
   
    if opt.model == 'gas':
        model = GAS(opt)
        nn.init.xavier_uniform_(model.conv1.weight)
        nn.init.xavier_uniform_(model.conv2.weight)
        nn.init.xavier_uniform_(model.fc.weight)

    elif opt.model == 'resnet18': 
        print("=> using pre-trained model '{}'".format(opt.model))
        model = torch_models.__dict__[opt.model](pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)
        nn.init.xavier_uniform_(model.fc.weight)

    elif opt.model == 'inception_v3': 
        print("=> using pre-trained model '{}'".format(opt.model))
        model = torch_models.__dict__[opt.model](pretrained=True)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
    
    elif opt.model == 'GA-Inception' or opt.model == 'GA-Resnet':
        model = load_glp(opt)

    else:
        raise ValueError('the model is not supported') 

    return model