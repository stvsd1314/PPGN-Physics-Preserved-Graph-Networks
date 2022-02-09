# PPGN:Physics-Preserved-Graph-Networks  
============================================================================== 
This software is to locate faults in distribution systems with limited observations and labels through PPGN. PPGN performs better than the baselines when labeled data are insufficient and the distribution of data vary randomly.   Code accompanying the paper ["PPGN: Physics-Preserved Graph Networks for Fault Location with Limited Observation and Labels"] 

## Prerequisites
The proposed method is implemented through Jupyter Notebook. The required packages include:
- Jupyter Notebook
- Python 3
- Python packages: Numpy, torch, time, os, scipy, matplotlib

## Getting started
1) You can train the proposed model with "training_123nodes.ipynb" and the test the well-trained model through "Testing_123nodes.ipynb" for the IEEE 123-node test case. 
2) Similarly, our method is easily extended in IEEE 37-node test case. The training model is in "Training_37nodes.ipynb".
3) data: this folder contains the training and testing datasets in various situations. 
4) trained: this folder has the pre-trained models, including the proposed and the baselines. 
 
 

 
 
