import os
import csv
import json
import time
import torch
import argparse

import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

from tqdm import tqdm
from PIL import Image
from torchvision import models
from torchsummary import summary
from torchvision import transforms, utils
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score




class tow_stream_net(nn.Module):
    def __init__(self, pre_model, glp_model):
        super().__init__()
        self.fc = nn.Linear(52224, 1000)
        self.pre_model = pre_model
        self.glp_model = glp_model
        
        self.pre_model.eval()
        self.glp_model.eval()

    def forward(self,x):
        out1 = self.pre_model(x)
        out1 = torch.flatten(out1, 1)
        out2 = self.glp_model(x)
        # print(out1.shape , out2.shape)
        out = self.fc(torch.cat((out1,out2),1))
        return out




class GLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5,stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(16384, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.batchnorm(F.relu(self.pool(self.conv2(self.conv1(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        return x




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


def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fscore = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        # print(data)
        input = data[0].cuda()
        target = data[1].squeeze(-1).long().cuda()
        val_loader_len = len(val_loader.dataset)

        # compute output
        with torch.no_grad():
            output = model(input)

        # measure accuracy and record loss
        prec1, prec5, f1score, precision, recall = accuracy(output.data, target, topk=(1, 5))

        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        fscore.update(f1score, input.size(0))
        precisions.update(precision, input.size(0))
        recalls.update(recall, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Top@1 {top1.avg:.3f} Top@5 {top5.avg:.3f} F1-Score {fscore.avg:.3f} \
            precision {precisions.avg:.3f} recall {recalls.avg:.3f} '
        .format(top1=top1, top5=top5, fscore=fscore, precisions=precisions , recalls=recalls))


def train(model, train_loader, test_loader,device):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    min_valid_loss = np.inf

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        model.train()
        print(f'start epoch {epoch+1}')
        # print(train_loader)
        for data in tqdm(train_loader):
            # print(i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels= labels.to(device)
            #print(len(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
           
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        print("validation Mode!")
        correct = 0
        total = 0
        validation_loss =0
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images = images.to(device)
                labels= labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                valloss = criterion(outputs, labels)
                validation_loss += valloss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the train images: {(100 * correct_train / total_train):.2f}')
        print(f'Accuracy of the network on the validation images: {(100 * correct / total):.2f}')
        print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {validation_loss / len(test_loader)}')
        if min_valid_loss > validation_loss/ len(test_loader):
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{validation_loss/len(test_loader):.6f}) \t Saving The Model')
            min_valid_loss = validation_loss/len(test_loader)
            # Saving State Dict
            torch.save(model.state_dict(), 'final_saved_model_ep{}.pth'.format(epoch))

    print('Finished Training')

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
  
    args = parser.parse_args()
    return args


def main():

    args = parse()

    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")

    transform= transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop(224),              
                                    transforms.ToTensor(),                     
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])])

    data_train = datasets.ImageNet(root= args.data[1], split = 'train',
                       transform=transform)
    data_test = datasets.ImageNet(root= args.data[1], split = 'val',
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    print("=> using pre-trained model '{}'".format(args.arch))
    pre_model = models.__dict__[args.arch](pretrained=True)
    #freezing all layers: 
    for param in pre_model.parameters():
        param.requires_grad = False
    pre_model_conv = nn.Sequential(*list(pre_model.children())[:-1]) #for resnet
    pre_model_conv = pre_model_conv.to(device)
   
    glp_model = GLP()
    #load pretrain model
    glp_path='saved_model_ep3-4002_3264.pth'
    glp_model.load_state_dict(torch.load(glp_path))
    #freezing all layers: 
    for param in glp_model.parameters():
        param.requires_grad = False
        
    glp_model = glp_model.cuda(1)
    # print(summary(glp_model,(3,224,224)))

    train_model = tow_stream_net(pre_model_conv,glp_model)
    train_model=train_model.to(device)
    # print(summary(train_model,input_size=(3,224,224)))
    
    train(train_model, train_loader, test_loader, device)



if __name__ == '__main__':
    main()