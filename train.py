
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



def train(opt, data):


    min_valid_loss = np.inf

    if opt.model == 'glp':
        model = GLP(opt)
        torch.nn.init.xavier_uniform_(model.conv1.weight)
        torch.nn.init.xavier_uniform_(model.conv2.weight)
        torch.nn.init.xavier_uniform_(model.fc.weight)

    elif opt.model == 'resnet18': 
        print("=> using pre-trained model '{}'".format(opt.model))
        model = models.__dict__[opt.model](pretrained=True)
        for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)
        torch.nn.init.xavier_uniform_(model.fc.weight)

    elif opt.model == 'inception_v3': 
        print("=> using pre-trained model '{}'".format(opt.model))
        model = models.__dict__[opt.model](pretrained=True)
        for param in model.parameters():
                param.requires_grad = False

        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        torch.nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, opt.num_classes)
        torch.nn.init.xavier_uniform_(model.fc.weight)


    if opt.resume : 
        load_path = os.path.join(opt.outf, opt.name, 'checkpoints')
        model.load_state_dict(torch.load(opt.checkpoint_path, map_location='gpu'))

    model = model.cuda(1)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    for epoch in range(opt.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        model.train()

        print(f'start epoch {epoch+1}')
        for images, labels in tqdm(data.train):

            images = images.cuda(1)
            labels= labels.cuda(1)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if opt.model == 'inception_v3':

                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else: 
                outputs = model(images)
                loss = criterion(outputs, labels)
           
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        scheduler.step()

        print("validation Mode!")
        correct = 0
        total = 0
        validation_loss =0
        model.eval()

        with torch.no_grad():
            tic = time.time()
            for images, labels  in tqdm(data.valid):

                images = images.cuda(1)
                labels= labels.cuda(1)

                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                valloss = criterion(outputs, labels)

                validation_loss += valloss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        save_path = os.path.join(opt.outf, opt.name, 'checkpoints')
        print( f'AVG validation time: {(time.time()-tic)/len(data.valid.dataset)}')
        print(f'Accuracy of the network on the train images: \
                    {(100 * correct_train / total_train):.2f}')
        print(f'Accuracy of the network on the validation images:\
                    {(100 * correct / total):.2f}')
        print(f'Epoch {epoch+1} \t\t Training Loss: \
                    {running_loss / len(data.train.dataset)} \t\t \
                    Validation Loss: {validation_loss / len(data.valid.dataset)}')
        if min_valid_loss > validation_loss/ len(data.valid.dataset):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{validation_loss/len(data.valid.dataset):.6f}) \
                    \t Saving The Model')
            min_valid_loss = validation_loss/len(data.valid.dataset)
            # Saving State Dict
            torch.save(model.state_dict(), os.path.join(save_path,opt.model + f'_ep{epoch+1}_acc{(100 * correct / total):.2f}.pth'))
            best_model = model

    print('Finished Training')
    return best_model