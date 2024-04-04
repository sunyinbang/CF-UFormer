## Low-Light Image Enhancement using Transformer with Color Fusion and Channel Attention  

## [YinBang Sun]((https://github.com/sunyinbang)), Jing Sun, Fuming Sun,  Fasheng Wang, Haojie Li  

>**Abstract:** Low-Light Image Enhancement (LLIE) aims to optimize images captured in low-light conditions with low brightness and contrast, rendering them natural-looking images that are more aligned with the human visual system. However, existing methods could not simultaneously solve the problems of color distortion, noise amplification and loss of details during the enhancement process. To this end, we propose a novel low-light image enhancement network, referred to as U-shape Transformer with Color Fusion (CF-UFormer), which employs Transformer Block as its fundamental element and comprises three modules: Feature Extraction Module (FEM), U-Former structure, and Refinement Module (RM). Firstly, FEM leverages three color spaces with different color gamuts to extract shallow features, thus retaining a wealth of color and detail information in the enhanced image. In addition, we take the Channel Attention Mechanism (CAM) to the U-Former structure to compensate for the lack of spatial dimension information interaction, which can suppress the noise amplification caused by continuous downsampling through adaptively learning the weight parameters between channels. Finally, to deal with the single expression ability of the $ {L_{1}} $ loss function used in most existing methods, CF-UFormer selects four loss functions to train on the LOL dataset, resulting in excellent qualitative and quantitative evaluation criteria on various benchmark datasets. The codes and models are available at https://github.com/sunyinbang/CF-UFormer.

This repository contains the dataset, code and pre-trained models for our paper.

## Network Architecture
![CF-UFormer](figures/CF-UFormer.jpg)

We propose a new method for Low-Light Image Enhancement. CF-UFormer is an end-to-end deep auto-encoder structure based on Transformer. It consists of Feature Extraction Module (FEM), a U-Former structure and Refinement Module (RM). FEM converts RGB images to the HSV and LAB colour spaces and fusion them with Multi-level Fusion Unit (MFU) to generate more realistic colors. The U-Former structure with a improved Channel Attention Mechanism (CAM) focuses on more important channel dimensions, making up for the shortcomings of the Transformer Block, which only calculates length and width attention scores. RM effectively restores the texture, edge and other image detail information.

## Get Started

### Dependencies and Installation
1. Create Conda Environment 
```
conda create -n CF_UFormer python=3.7
conda activate CF_UFormer
conda install pytorch=1.8 torchvision=0.3 cudatoolkit=10.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```
2. Clone Repo
```
git clone https://github.com/sunyinbang/CF-UFormer.git
```

3. Install warmup scheduler

```
cd CF_UFormer
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

### Dataset
You can use the following links to download the datasets

1. LOL [[Link](https://daooshee.github.io/BMVC2018website/)]
2. LSRW [[Link](https://github.com/abcdef2000/R2RNet )]

### Pretrained Model
We provide the pre-trained models under different datasets:
- CF-UFormer trained on LOL [[Baidu drive](https://pan.baidu.com/s/1UVwHCj-bpJ1b4mPDJUHAvg?pwd=1234)] with training config file `./configs/LOL/train/training_LOL.yaml`.
- CF-UFormer trained on LSRW [[Baidu drive](https://pan.baidu.com/s/1sYq0hCGPk5hLe5f2qfVmAA?pwd=1234)] with training config file `./configs/LSRW/train/training_LSRW.yaml`.


### Test
You can directly test the pre-trained model as follows

1. Modify the paths to dataset and pre-trained mode. 
```python
# Tesing parameter 
input_dir # the path of data
result_dir # the save path of results 
weights # the weight path of the pre-trained model
```

2. Test the models for LOL and LSRW dataset

You need to specify the data path ```input_dir```, ```result_dir```, and model path ```weight_path```. Then run
```bash
python test.py --input_dir your_data_path --result_dir your_save_path --weights weight_path
```

### Train

1. To download LOL/LSRW training and testing data

3. To train CF-UFormer, run
```bash
python train.py --opt your_config_path
```
```
You need to modify the config for your own training environment.
```

## Quantitative results

### Results onLOL

![LOL](figures/LOL.jpg)

### Results on LSRW

![LSRW_Huawei](figures/LSRW_Huawei.jpg)

![LSRW_Nikon](figures/LSRW_Nikon.jpg)

## Qualitative results

### Results on LOL

![wardrobe](figures/wardrobe.jpg)

### Results on MEF

![MEF](C:\Users\Administrator\Desktop\论文图像\MEF.jpg)

## Contact

If you have any questions, please contact kevinbang@126.com

---




