
import os
import time
import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from options import Options

from lib.models import load_model
from lib.utils import seed_everything
from lib.data.dataloader import load_data




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(opt, data, model):

    

    criterion = nn.CrossEntropyLoss()

    print("Testing Mode!")
    model.eval()
    if 'navon' in opt.dataset:
        with torch.no_grad():
            correct = 0
            total = 0
            tic = time.time()
            print('test on novon local')
            for images, labels  in tqdm(data.test[0]):

                images = images.to(device)#.cuda(0)
                labels= labels.to(device)#.cuda(0)

                outputs = model(images)
 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'len test{len(data.test[0].dataset)}')
            print( f'AVG test time: {(time.time()-tic)/len(data.test[0].dataset)}')
            print(f'Accuracy of the network on the test images:\
                        {(100 * correct / total):.2f}')
            

            correct = 0
            total = 0
            tic = time.time()
            print('test on novon global')
            for images, labels  in tqdm(data.test[1]):

                images = images.to(device)#.cuda(0)
                labels= labels.to(device)#.cuda(0)

                outputs = model(images)
 
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'len test{len(data.test[1].dataset)}')
            print( f'AVG test time: {(time.time()-tic)/len(data.test[1].dataset)}')
            print(f'Accuracy of the network on the test images:\
                        {(100 * correct / total):.2f}')

            
        print('Finished Testing')

    else:

        with torch.no_grad():
            correct = 0
            total = 0
            tic = time.time()
            for images, labels  in tqdm(data.test):

                images = images.to(device)
                labels= labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'len test{len(data.test.dataset)}')
        print( f'AVG test time: {(time.time()-tic)/len(data.test.dataset)}')
        print(f'Accuracy of the network on the test images:\
                    {(100 * correct / total):.2f}')

        print('Finished Testing')


if __name__ == '__main__': 

    opt = Options().parse()

    '''SEED Everything'''
    seed_everything(SEED=opt.seed)


    data = load_data(opt)
    model = load_model(opt)
    model.load_state_dict(torch.load(opt.checkpoint_path, map_location='cpu'))
    model = model.to(device)


    test(opt, data, model)