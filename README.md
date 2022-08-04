# image_classification
## Desription
This git files are image classification neural network models (VGG, Inception, ResNet) using pytorch. 

## Installation
### conda setting
The conda folder has environment yaml file. 
  conda env create -n {environment name} -file image_classification.yaml
### pip file
I would not release the docker environment file but you can use the pytorch docker for this codes. 
'''
docker pull pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
'''
And I would recommend the pytorch version 1.8.
Packages are in requirement.txt. If you want pip installation, you can use pip requirement file.
'''
pip install -r requirements.txt
'''

## Execution
For the training
'''
  python train.py --model_type ResNet50 --dataset STL10 --lr 0.001 --running_lr False --epochs 100 --data_dir ./data --checkpoint ./checkpoint
  --log_dir ./logs --aux_layer False
'''
If you want to use inceptionV1, inceptionV2, inceptionV3, you would choose the application of aux layers. So, you should check the --aux_layer first.

For the test
'''
  python test.py
'''
