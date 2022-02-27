"""
Module responsible for data augmentation constants and configuration.
"""

import math
import torch as ch

from torchvision import transforms
from PIL import Image , ImageFilter



def image_entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])


class add_gussianblur(object):
    def __init__(self):
       self.x=3
    def __call__(self, image):
        IDX=[]
        sz=image.size
        # print(sz[0])
        ENT=[]
        for i in range(1,int(sz[0]/4)):
           image_filtered = image.filter(ImageFilter.GaussianBlur(radius=i))                        
           ENT.append(image_entropy(image_filtered))   
           # print("ENT",ENT)
        #print(ENT)
        idx=ENT.index(max(ENT))  
        #print(idx)
        IDX.append(idx)
        image_filtered = image.filter(ImageFilter.GaussianBlur(radius=idx)) 
        #imshow(np.asarray(image_filtered))
        # image_filtered.save('SimpleShapeDB_filtered/'+p[0])
        return image_filtered
        


# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# Special transforms for ImageNet(s)
# TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(
#             brightness=0.1,
#             contrast=0.1,
#             saturation=0.1
#         ),
#         transforms.ToTensor(),
#         Lighting(0.05, IMAGENET_PCA['eigval'], 
#                       IMAGENET_PCA['eigvec'])
#     ])
TRAIN_TRANSFORMS_IMAGENET=transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    add_gussianblur(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 
])
"""
Standard training data augmentation for ImageNet-scale datasets: Random crop,
Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
"""

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        add_gussianblur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
"""
Standard test data processing (no augmentation) for ImageNet-scale datasets,
Resized to 256x256 then center cropped to 224x224.
"""

# Data Augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""
