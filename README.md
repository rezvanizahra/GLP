# Global-Local Processing

<!-- Plz check the results [here](https://docs.google.com/spreadsheets/d/13QsrJIkUS6I4momK1m8u8F-cTiHTcXKn7WgOHLTggyw/edit#gid=0) -->

This repository contains Pytorch implementation for training and evaluating of the following paper: 

Global-Local Processing in Convolutional Neural Networks #[[1]](#6-reference)

<!-- <img src="Images/Deep-Disaster_all-min.png" width="600" height ="400"/>   -->
<!-- <img src="Images/Deep-Disaster_model_define-min.png" width="400" height="300"/> -->


## Installation
1. clone this repository
   ```
   git clone https://github.com/rezvanizahra/GLP.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n GLP_env python=3.7
    ```
3. Activate the virtual environment.
    ```
    conda activate GLP_env
    ```
4. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
## Dataset:
This repository employ disaster detection on Navon and Caltech101 datasets.
To use this dataset: 
<!--   1. Download from [here](https://crisisnlp.qcri.org/data/ASONAM17_damage_images/ASONAM17_Damage_Image_Dataset.tar.gz)
  2. Unpack it into the `data` folder.  
  3. Prepare data such as this file structure:  -->

To train the model on your custom dataset, you could copy the dataset into `./data` directory or set the --dataroot argument. 

## Training and Testing:
To list the training parameters, run the following command:
```
python main.py -h
```
### To train the model:
``` 
python main.py  --model <model_name>  --dataset <dataset_name> --epochs <number-of-epochs> --num_classes <number_of_classes> --add_smartfilter 
```
### To test the model:
``` 
python test.py  --model <model_name>  --dataset <dataset_name> --num_classes <number_of_classes>  --checkpoint_path  <path
_to_checkpoint>                                            
```
### To test the model on attacks and save results in a .csv file:
``` 
python test_on_attacks.py  --model <model_name>  --dataset <dataset_name> --num_classes <number_of_classes>  --checkpoint_path  <path
_to_checkpoint>  --attack_method <FGSM | PGD>  --repeat_on_attacks <number-of_repeat> --save_csv <path_to_save_csv_file>                                    
```
####note: 
To train and test GA-Resnet and GA-Inception you need to define the --gas-path which defined the path to the gas model checkpoints

<!-- ## Citating Deep-Disaster
If you want to cite this work in your publication:
``` bash
@misc{shekarizadeh2022deepdisaster,
      title={Deep-Disaster: Unsupervised Disaster Detection and Localization Using Visual Data}, 
      author={Soroor Shekarizadeh and Razieh Rastgoo and Saif Al-Kuwari and Mohammad Sabokrou},
      year={2022},
      eprint={2202.00050},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` -->
## Reference
<!-- [1] [Deep-Disaster: Unsupervised Disaster Detection and Localization Using Visual Data](https://arxiv.org/pdf/2202.00050.pdf). -->

