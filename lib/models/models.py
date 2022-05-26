import torch
import torch.nn as nn
import torch.nn.functional as F


GLP_fc_size = { 'GA-Inception':315904,  'GA-Resnet':125440 } #'GA-Inception':306304
GAS_fc_size = { 299:175232,  224:100352 }


class GLP(nn.Module):
    """
    Global Local Processing Model
    
    """
    def __init__(self, opt, pre_model, gas_model):
        super().__init__()

        self.pre_model = pre_model
        self.gas_model = gas_model
        self.dropout = nn.Dropout(p=0.2)
        if opt.model in GLP_fc_size.keys():
            self.fc = nn.Linear(GLP_fc_size[opt.model], opt.num_classes)
        else:
            raise ValueError('please defined the size of fc layer for GLP model')


    def forward(self,x,is_inception=False):
        if is_inception:
            out1, _ = self.pre_model(x)
        else:
            out1= self.pre_model(x)
        out1 = torch.flatten(out1, 1)
        out2 = self.gas_model(x)
        out2 = torch.flatten(out2, 1)
        out = self.fc(self.dropout(F.relu(torch.cat((out1,out2),1))))
        return out


class GAS(nn.Module):
    """
    Global Advantage Stream Model

    """
    def __init__(self,opt):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5,stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.batchnorm = nn.BatchNorm2d(128)
        if opt.icrop in GAS_fc_size.keys():
            self.fc = nn.Linear(GAS_fc_size[opt.icrop], opt.num_classes)
        else:
            raise ValueError('please defined the size of fc layer for GAS model')
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.batchnorm(F.relu(self.pool(self.conv2(self.conv1(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc(self.dropout(x)))
        return x
