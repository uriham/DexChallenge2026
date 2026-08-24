#!/bin/bash

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
pip install -e .
cd ..

cd pytorch_kinematics
pip install -e .
cd ..

pip install pip==23.3.1

pip install -e .
