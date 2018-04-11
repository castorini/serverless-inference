#!/bin/bash

# This is a script to craete deployment package for running PyTorch model on AWS Lambda.
# It's suggested to use this script on an EC2 instance with at least 2 vCPU and 4 GB of RAM.
# To use this script, simply put it in home directory of the EC2 instance and run it, it will
# create a `deployment_package.zip` which is the deployment pacakge for AWS Lambda.
#
# You also need to manually put into it your PyTorch models and files and then upload the package to S3, and 
# configure your Lambda to load the package from there.

# update machine and install necessary libs/programs
sudo yum -y update
sudo yum install -y gcc gcc-gfortran gcc-c++ zlib zlib-devel openssl openssl-devel git make automake kernel-devel
sudo yum install -y openblas-devel

# cp OpenBLAS libs and dependencies to /var/task to simulate Lambda's runtime environment
sudo mkdir /var/task
sudo find /usr/lib64 -name "libopenblas.*" -exec cp {} /var/task \;
sudo find /usr/lib64 -name "libgfortran.*" -exec cp {} /var/task \;
sudo find /usr/lib64 -name "libquadmath.*" -exec cp {} /var/task \;

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

# numpy dependency
pip3 install cython
# pytorch dependency
pip3 install pyyaml  


# install numpy and pytorch from source to reduce package size
cd
git clone --recursive https://github.com/numpy/numpy.git
cd numpy
# use openblas
echo "[openblas]" >> ./site.cfg
echo "libraries = openblas" >> ./site.cfg
echo "library_dirs = /var/task" >> ./site.cfg
echo "include_dirs = /var/task" >> ./site.cfg
python3 setup.py config
python3 setup.py install

# and pytorch...
cd
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
export NO_CUDA=1  # don't need CUDA for inferencing
export NO_CUDNN=1 # don't need CUDNN for inferencing
python3 setup.py install

# create the deployment package
cd $VIRTUAL_ENV/lib/python3.6/site-packages
zip -r9 ~/deployment_package.zip *
cd /var/task
zip -r9 ~/deployment_package.zip *
