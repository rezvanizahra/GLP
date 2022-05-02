import torch
import torch.nn as nn
import torch.nn.functional as F


class TWO_STREAM(nn.Module):
    def __init__(self, pre_model, glp_model):
        super().__init__()
        self.fc = nn.Linear(125440, opt.num_classes) #177280 inception #125440 resnet18
        self.pre_model = pre_model
        self.glp_model = glp_model
        self.dropout = nn.Dropout(p=0.2)


    def forward(self,x,is_inception=False):
        if is_inception:
            out1, _ = self.pre_model(x)
        else:
            out1= self.pre_model(x)
        out1 = torch.flatten(out1, 1)
        out2 = self.glp_model(x)
        # print(out1.shape , out2.shape)
        out = self.fc(self.dropout(F.relu(torch.cat((out1,out2),1))))
        return out



class GLP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5,stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.batchnorm = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(100352, opt.num_classes) #32768 128 #224 100352 #299 175232
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.batchnorm(F.relu(self.pool(self.conv2(self.conv1(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)

        return x

