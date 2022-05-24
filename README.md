# Global-Local Processing

<!-- Plz check the results [here](https://docs.google.com/spreadsheets/d/13QsrJIkUS6I4momK1m8u8F-cTiHTcXKn7WgOHLTggyw/edit#gid=0) -->

This repository contains Pytorch implementation for training and evaluating of the following paper: 

Global-Local Processing in Convolutional Neural Networks [[1]](#6-reference)

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
### To train the model on the Caltech101 dataset:
``` 
python main.py --dataset <dataset_name> --niter <number-of-epochs>                                                
```
### To test the model on the Caltech101 dataset:
``` 
python test.py --dataset <dataset_name> --load_weights                                                 
```

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

