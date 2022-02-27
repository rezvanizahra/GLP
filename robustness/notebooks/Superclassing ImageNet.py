in_path = '../../ILSVRC/Data/CLS-LOC/'
in_info_path = '../imagenet_info'

from robustness.tools.imagenet_helpers import ImageNetHierarchy
in_hier = ImageNetHierarchy(in_path,
                            in_info_path)

for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
    if cnt < 10:
        print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")

# ancestor_wnid = 'n00001740'
# print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")
# for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
#     if wnid in in_hier.in_wnids:
#         print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")



n_classes = 9

superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_classes,
                                           ancestor_wnid='n00001740',
                                           # superclass_lowest=['n02084071'],
                                           balanced=False)


labels_unique = []
for label in class_ranges: 
	labels_unique.extend(label)

print("unique labels",len(labels_unique))

from robustness.tools.imagenet_helpers import common_superclass_wnid
import torchvision.datasets as dsets
from torchvision import transforms
import torch

# superclass_wnid = common_superclass_wnid('big_12')
# class_ranges, label_map = in_hier.get_subclasses(superclass_wnid,
#                                                  balanced=False)
# print(class_ranges)
# print(label_map)
# labels_unique = []
# for label in class_ranges: 
# 	labels_unique.extend(label)

# print("unique labels",len(labels_unique))

from robustness import datasets

custom_dataset = datasets.CustomImageNet(in_path,
                                         class_ranges)

train_loader, test_loader = custom_dataset.make_loaders(workers=10,
                                                        batch_size=5)

print(f"Train set size: {len(train_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")

val_transforms= transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),              
                                    transforms.ToTensor(),                     
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])])

data_test = dsets.ImageNet(root= '../../ILSVRC/Data/CLS-LOC/', split = 'val',
                       transform=val_transforms)
data_train = dsets.ImageNet(root= '../../ILSVRC/Data/CLS-LOC/', split = 'train',
                       transform=val_transforms)

test_ldr = torch.utils.data.DataLoader(
        data_test,
        batch_size=5, shuffle=False,
        num_workers=10, pin_memory=True)

train_ldr = torch.utils.data.DataLoader(
        data_train,
        batch_size=5, shuffle=False,
        num_workers=10, pin_memory=True)

print(f"Train set size: {len(train_ldr.dataset)}")
print(f"Test set size: {len(test_ldr.dataset)}")