# Scarf: Soil Carbon Sensing with Wi-Fi and Optical Signals
This repository contains the source code and data for the MobiCom'24 paper Scarf.

## Environment Requirements
- Matlab: we use MATLAB_R2022a. 
- A Linux machine with a GPU that is able to run ResNet is required. We use Linux 22.04 and CUDA version 12.3. We have tested NVIDIA GTX 1080, RTX 4070 and RTX 4090.
- The version of required Python packages we use are listed in requirements.txt
- Install required Python packages

```shell
chmod +x setup.sh
./setup.sh

```
## Running Instructions
### Matlab
- The Matlab scripts are located in `matlab/`. Each script corresponds to a figure in the paper. To execute a script, open it in Matlab and run all the sections to get the figures. 
- We provide pre-processed data (permittivity, lightness, and oven-based volumetric water content) associated with the scripts.
- The scripts contains the mathematical models described in the papar.

### Pytorch
- The Pytorch scripts are located in `resnet/`. 
- We provide representative pre-trained ResNet models for Table 4 in the paper. Specifically, we provide a model for only using machine learning and another model for using a combination of mathematical model and machine learning. The models are located in `resnet/pretrained_models/`.
- We provide the testing dataset for evaluating the pre-trained models. The dataset is in `resnet/testing_data/`.
- To evaluate the performance of the pre-trained models, run the following:

```shell
cd resnet
python3 run_ml_model_test.py
python3 run_math_and_ml_model_test.py
```



 
