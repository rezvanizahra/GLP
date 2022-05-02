
import os
import time
import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import models

from lib.models.GLP import GLP



def test(opt, data, model):

    criterion = nn.CrossEntropyLoss()

    print("Testing Mode!")
    correct = 0
    total = 0
    loss =0
    model.eval()

    with torch.no_grad():
        tic = time.time()
        for images, labels  in tqdm(data.test):

            images = images.cuda(1)
            labels= labels.cuda(1)

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            test_loss = criterion(outputs, labels)

            loss += test_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'len test{len(data.test.dataset)}')
    print( f'AVG test time: {(time.time()-tic)/len(data.test.dataset)}')
    print(f'Accuracy of the network on the test images:\
                {(100 * correct / total):.2f}')

    print('Finished Testing')
