# Pruning-aware merging for multitask inference

This repository contains code of the paper, [Pruning-aware merging for multitask inference (KDD 2021)]. 
It includes all codes for training, pruning and validation. 


### Dependencies

This code requires

 - python 3.*
 - TensorFlow v1.13

### Data

The dataset in this repository includes CelebA, LFW, Deepfashion, FashionMNIST and Cifar10/100. The details are in ```config/data.cfg``` and ```config/global.cfg```, and you need to change it to your own path.

### Model

The models includes VGG72/128/512, ResNet18/34, and Lenet5. Details are in ```config/model.cfg```(The terms begin with ```rdnet_``` is the config for multi-task learning). 

### Usage

To run the code for training and pruning, see the usage instructions in `exp_run`.

For general CNN:
- train_model.py: Train or prune CNN by [1]
- train_rdnet.py: Train or prune a multitask model by [1]
- val_model.py: Verify CNN on specific dataset and task
- val_rdnet.py: Verify multitask model on specific dataset and task
- prune_cvpr.py: Prune CNN by [2]
- prune_cvpr_multi.py: Prune multitask model by [2]
- prune_res_cvpr.py: Prune residual network by [2]

For residual network:
- train_resnet.py: Train a residual network
- train_rdnet_res.py: Similar with ```train_rdnet.py```, but for ResNet18/34

### On Going
- Clean code
- Extract all paths into ```config```

### Reference
[1] Bin Dai, Chen Zhu, Baining Guo, and David Wipf. 2018. Compressing neural networks using the variational information bottleneck. In ICML. ACM, New York, NY, USA, 1143–1152.
[2] PavloMolchanov,ArunMallya,StephenTyree,IuriFrosio,andJanKautz.2019. Importance estimation for neural network pruning. In CVPR. IEEE Press, Piscataway, NJ, USA, 11264–11272.