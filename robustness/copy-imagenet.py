import os
import shutil
import random
from tqdm import tqdm



cpy_dir = '../ILSVRC_tiny/val'
root_dir = '../ILSVRC/Data/CLS-LOC/val'

for f in tqdm(os.listdir(root_dir)):
    class_path = os.path.join(root_dir,f)
    all_files = [name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))]
    print(f'class_name: {f},all_files: {len(all_files)}')
    copy_files = random.sample(all_files, int(0.1*int(len(all_files))))
    print(f'just 10 percent: {len(copy_files)}')
    
    copy_dest = os.path.join(cpy_dir, class_path)
    if not os.path.exists(copy_dest):
        os.makedirs(copy_dest)
        print('class dir created!')

        for file in tqdm(copy_files):  
            shutil.copy(os.path.join(class_path,file), os.path.join(copy_dest,file))
  