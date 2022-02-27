in_path = '../ILSVRC/Data/CLS-LOC/'
in_info_path = 'imagenet_info'

from robustness.tools.imagenet_helpers import ImageNetHierarchy
in_hier = ImageNetHierarchy(in_path,
                            in_info_path)

# for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
    # if cnt < 10:
    # if wnid in ['n00002137', 'n00001930', 'n04424418']:
    # if wnid in ['n00020827', 'n00029677', 'n14580597', 'n00002452','n00007347', 'n00002684']:
    # if wnid in ['n00033615', 'n07999699', 'n00024264', 'n00031264','n00023100', 'n00033020', 'n00031921', 'n05810143']:
    # if wnid in ['n13873917', 'n13861050', 'n13900422', 'n13863771', 'n13862780', 'n13865483', 'n13862644', 'n13879634', 'n13860793', 'n13862282', 'n13900760', 'n13870805', 'n13878951', 'n13867492', 'n13864763', 'n13867276']:
    # if wnid in ['n09287968', 'n03532080', 'n09432990', 'n09407346', 'n09300905', 'n09251689', 'n03610270', 'n09474162',\
    #              'n04345288', 'n04012260', 'n09238143', 'n03233423', 'n09358550', 'n09295338', 'n07851054', 'n09281777', \
    #              # 'n09308398', 'n09283193', 'n03595179', 'n03714721', 'n09409203', 'n09335240', 'n00027167', 'n03149951',\
    #               'n03009633', 'n04248010', 'n04486445', 'n09267490', 'n00003553', 'n09302031', 'n09334396', 'n03338648', \
    #               'n09368224', 'n03892891', 'n09468237', 'n09279458', 'n09477037']:
        # print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")

# ancestor_wnid = 'n00002137'
# print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")
# for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
    # if wnid in in_hier.in_wnids:
        # print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")



n_classes = 2

superclass_wnid, class_ranges, label_map ,names= in_hier.get_superclasses(n_classes,
                                           ancestor_wnid='n00001740',
                                           superclass_lowest=['n00031921', 'n00033020','n00024264','n00002684'\
                                                               , 'n00007347'],
                                           balanced=False)

print(superclass_wnid, names)
labels_unique = []
for label in class_ranges: 
	labels_unique.extend(label)

print("unique labels",len(labels_unique))

n_classes = 2

superclass_wnid, class_ranges, label_map , names= in_hier.get_superclasses(n_classes,
                                           ancestor_wnid='n00001740',
                                           # superclass_lowest=['n00031921', 'n00033020','n00024264','n00002684'\
                                                               # 'n00020827', 'n00007347'],
                                           balanced=False)

print(superclass_wnid, names)
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

data_test = dsets.ImageNet(root= '../ILSVRC/Data/CLS-LOC/', split = 'val',
                       transform=val_transforms)
data_train = dsets.ImageNet(root= '../ILSVRC/Data/CLS-LOC/', split = 'train',
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