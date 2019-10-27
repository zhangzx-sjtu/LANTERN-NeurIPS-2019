# LANTERN-NeurIPS-2019
Source code for NeurIPS 2019 paper "Learning Latent Processes from High-Dimensional Event Sequences via Efficient Sampling"
## Environment
+ Python 3.5
+ PyTorch 1.0.1
## Requirements
+ GPUs with 12GB memory
## Datasets
+ The memetracker dataset can be downloaded from：https://snap.stanford.edu/data/memetracker9.html
+ The weibo dataset be doenloaded from: https://www.aminer.cn/influencelocality
+ Our great thanks to authors of the datasets.
+ Use data/#dataset#/preprocess.py to preprocess the downloaded dataset and you can get the .pkl files in each folder
## Quick Start
To train on small datasets (Syn-Small and Memetracker), you can run
```
python train_small.py
```
To train on large datasets (Syn-Large and Weibo), you can run
```
python train_large.py
```
We also released our pre-trained model parameters for each dataset in /model folder. For a quick test, run
```
python test.py
```
## Citation
If you have any problems on this code, feel free to contact zhangzx369@gmail.com.
If you use this code as part of your research, please cite the following paper:
```
@inproceedings{LANTERN-19,
  author    = {Qitian Wu and Zixuan Zhang and Xiaofeng Gao and Junchi Yan and
               Guihai Chen},
  title     = {Learning Latent Process from High-DimensionalEvent Sequences via Efficient Sampling},
  booktitle = {Thirty-third Conference on Neural Information Processing Systems, {NeurIPS} 2019, Vancouver, Canada,
               Dec 8-14, 2019},
  year      = {2019}
  }
```
