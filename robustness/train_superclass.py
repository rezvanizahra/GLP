
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms, utils

from robustness import datasets
from robustness.tools.imagenet_helpers import ImageNetHierarchy

in_path = '../ILSVRC_tiny/ILSVRC/Data/CLS-LOC/'
in_info_path = 'imagenet_info'

in_hier = ImageNetHierarchy(in_path,
                            in_info_path)


n_classes = 10
superclass_wnid, class_ranges, label_map ,names= in_hier.get_superclasses(n_classes,
                                           ancestor_wnid='n00001740',
                                           # superclass_lowest=['n00031921', 'n00033020','n00024264','n00002684'\
                                           #                     , 'n00007347'],
                                           balanced=False)

print(superclass_wnid, names)
labels_unique = []
for label in class_ranges: 
	labels_unique.extend(label)

print("unique labels",len(labels_unique))


custom_dataset = datasets.CustomImageNet(in_path,class_ranges)
train_loader, test_loader = custom_dataset.make_loaders(workers=4, batch_size=128)

print(f"Train set size: {len(train_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5,stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2, padding=2)
        self.batchnorm = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(8192, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        #x = self.pool(x)
        # print(x.shape)
        x = self.batchnorm(F.relu(self.pool(self.conv2(self.conv1(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(self.dropout(x)))
        # x=self.conv1(x)
        # print(x.shape)
        return x

min_valid_loss = np.inf

net = Net()
net = net.cuda()
print(summary(net,(3,128,128)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    net.train()
    print(f'start epoch {epoch}')
    print(train_loader)
    for data in tqdm(train_loader):
        # print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels= labels.cuda()
        #print(len(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
       
        outputs = net(inputs)
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
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.cuda()
            labels= labels.cuda()
            # calculate outputs by running images through the network
            outputs = net(images)
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
        torch.save(net.state_dict(), 'filtered_saved_model_ep{}.pth'.format(epoch))

print('Finished Training')
