# LANTERN-NeurIPS-2019
Source code for NeurIPS 2019 paper "Learning Latent Processes from High-Dimensional Event Sequences via Efficient Sampling"
## Environment
+ Python 3.5
+ PyTorch 1.0.1
## Requirements
+ GPUs with at least 12GB memory
## Start
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
