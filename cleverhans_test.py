import os
import time
import torch
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torchvision.utils

import numpy as np
import torchvision.transforms as transforms

from easydict import EasyDict
from torchvision import models
import torchvision.datasets as dsets
from utils import image_folder_custom_label
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Numpy", np.__version__)

# print(sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name])))


class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  
])

imagnet_data = image_folder_custom_label(root='../../ILSVRC/Data/CLS-LOC/val', transform=transform, idx2label=idx2label)
# imagnet_data = dsets.ImageFolder(root='../../ILSVRC/Data/CLS-LOC/val', transform=transform)
# imagnet_data= dsets.ImageNet(root= '../../ILSVRC/Data/CLS-LOC/', split = 'val', transform=transform)
data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=8, shuffle=False)


use_cuda = True
device = torch.device("cuda:1" if use_cuda else "cpu")


# model = models.resnet50(pretrained=True).to(device)
# all_models=['resnet18','resnet101', 'vgg16', 'vgg19', 'mobilenet_v2','mobilenet_v3_large',
             # 'mnasnet1_0', 'efficientnet_b5', 'densenet121','squeezenet1_0'] #'resnet50', 'inception_v3' ,'alexnet'
all_models=['mobilenet_v2','mnasnet1_0', 
            'efficientnet_b5', 'densenet121','squeezenet1_0','inception_v3'] #vgg16','resnet50', 'inception_v3' ,'alexnet'
for model_name in all_models:
    print("=> using pre-trained model '{}'".format(model_name))
    model = models.__dict__[model_name](pretrained=True).to(device)

    model = model.eval()
    # epsilons = [.3]
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.15]
    for eps in epsilons:
        report = EasyDict(nb_test=0, correct_fgm=0)
        for x, y in tqdm(data_loader):
                x, y = x.to(device), y.to(device)
                x_fgm = fast_gradient_method(model, x, eps, np.inf)
                _, y_pred_fgm = model(x_fgm).max(
                    1
                )  # model prediction on FGM adversarial examples
                report.nb_test += y.size(0)
                report.correct_fgm += y_pred_fgm.eq(y).sum().item()

        print(
                f'test acc on FSGM eps {eps} : {report.correct_fgm / report.nb_test * 100.0 :.3f}'
                    
                )
            
       