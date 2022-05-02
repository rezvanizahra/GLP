import torch
import torch.nn as nn
import torch.nn.functional as F


#define glp network
class GLP(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5,stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.fc = nn.Linear(100352, opt.num_classes) #32768 128  #100352 224 #299  175232
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.batchnorm(F.relu(self.pool(self.conv2(self.conv1(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc(self.dropout(x)))
        return x

