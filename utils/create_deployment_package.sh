#!/bin/bash

# This is a script to craete deployment package for running PyTorch model on AWS Lambda.
# It's suggested to use this script on an EC2 instance with at least 2 vCPU and 4 GB of RAM.
# To use this script, simply put it in home directory of the EC2 instance and run it, it will
# create a `deploy.zip` which is the deployment pacakge for AWS Lambda. You can add into it
# your PyTorch models and files and then upload the package to S3, and configure your Lambda
# to load the package from there.

# update machine
sudo yum -y update
sudo yum install -y gcc zlib zlib-devel openssl openssl-devel git make automake gcc-c++ kernel-devel

# cmake 3.6.2
cd
wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
tar -zxvf cmake-3.6.2.tar.gz
cd cmake-3.6.2
sudo ./bootstrap --prefix=/usr/local
sudo make
sudo make install

# install python 3.6.1
cd
wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
tar -xzvf Python-3.6.1.tgz
cd Python-3.6.1 && ./configure && make
sudo make install

# setup virtual environment to install python packages
cd
sudo pip3 install virtualenv
virtualenv -p python3 ~/deploy
source ~/deploy/bin/activate

# pre-required packages
pip3 install cython  # numpy dependency
pip3 install pyyaml  # pytorch dependency

# install numpy and pytorch from source to reduce package size
cd
git clone --recursive https://github.com/numpy/numpy.git
cd numpy
python3 setup.py install

# and pytorch...
cd
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
export NO_CUDA=1  # don't need this for inferencing
export NO_CUDNN=1 # don't need this for inferencing
python3 setup.py install
pip3 install torchvision

# create the deployment package
cd $VIRTUAL_ENV/lib/python3.6/site-packages
zip -r9 ~/deploy.zip *