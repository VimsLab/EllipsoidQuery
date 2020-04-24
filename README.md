# EllipsoidQuery
A Novel Local Geometry Capture in Pointnet++ for 3D Classification

COMING SOON (June 2020)
-----------------------
PointNet++ with ellipsoid querying<br>

RS-CNN with ellipsoid querying<br>

This code submission is to reproduce the impact of reoriented ellipsoid querying on RS-CNN Shape Classification.<br>

Software requirements
----------------------
Ubuntu 18.04<br>
Python 3.5 (recommend Anaconda3)<br>
Pytorch 0.4.1<br>
CMake 3.10.2<br>
CUDA 10.0 + cuDNN 7<br>
Cudatoolkit V10.0.130<br>

Note: Also, works in the environment suggested by the authors of RS-CNN(https://github.com/Yochengliu/Relation-Shape-CNN/).<br>

Download
--------
git clone https://github.com/VimsLab/EllipsoidQuery.git<br>
cd EllipsoidQuery/RSCNNQE<br>

Building Kernel
---------------
mkdir build && cd build<br>
cmake .. && make<br>

Dataset
-------
Download and unzip ModelNet40 (415M) in data directory. <br>
https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip<br>

Usage: Train
------------
sh train_cls.sh<br>

Note: We have trained a Single-Scale-Neighborhood classification model in cls folder, whose training accuracy is 92.55% ('cls/model_cls_ssn_iter_70917_acc_0.925446.pth') <br>

Usage: Evaluation
-----------------
Modify cfgs/config_ssn_cls.yaml with *.pth file from cls/ folder with highest accuracy.<br>
python voting_evaluate_cls.py<br>

Note: You can use our model cls/model_cls_ssn_iter_70917_acc_0.925446.pth as the checkpoint in config_ssn_cls.yaml, and with majority voting you will get an accuracy of 93.51%<br>

This code has been heaviy borrowed from https://github.com/Yochengliu/Relation-Shape-CNN/ and https://github.com/erikwijmans/Pointnet2_PyTorch <br>


