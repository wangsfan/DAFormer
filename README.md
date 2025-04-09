# DAFormer
## Introduction
This is our implementation of our paper *Dual-Attention Transformers for Class-Incremental Learning: A Tale of Two Memories*. Authors: Shaofan Wang, Weixing Wang, Yanfeng Sun, Zhiyong Wang, Boyue Wang, Baocai Yin. Accepted by IEEE Transactions on Multimedia 

**TL;DR**: A dual attention mechanism for class-incremental learning.

**Abstract**:
Class-incremental learning (Class-IL) aims to continuously learn a model from a sequence of tasks, which suffers from the issue of catastrophic forgetting. Recently, a
few transformer based methods are proposed to address this issue by transferring self-attention into task-specific attention. However, these methods utilize shared task-specific attention modules across the whole incremental learning process, and are unable to achieve the balance between consolidation and plasticity, i.e., to remember the knowledge learned from previous tasks and absorb the knowledge from the current task simultaneously. Motivated by the mechanism of LSTM and hippocampus
memory, we point out that dual attention on long and shortterm memories can handle the consolidation-plasticity dilemma of Class-IL. Typically, we propose Dual-Attention Transformers (DAFormer) to learn external attention and internal attention. The former utilizes sample-dependent keys which exclusively focused on the new tasks, while the latter consolidates the knowledge from previous tasks by using sample-agnostic keys. We present two editions of DAFormer: DAFormer-S and DAFormer-M: the former utilizes shared external keys and maintains a small parameter size, while the latter utilizes multiple external keys and enhances the long-term memory. Furthermore, we propose the K-nearest neighbor invariant based distillation scheme, which distills knowledge from previous tasks to current task by
maintaining the same neighborhood relationship of each sample over old and new models. Experimental results on CIFAR-100, ImageNet-subset and ImageNet-full demonstrate that DAFormer significantly outperforms all the state-of-the-art parameter-static and parameter-growing methods. 

## Dependencies
- numpy==1.23.5
- torch==1.12.1
- torchvision==0.13.1
- timm==0.4.9
- continuum==1.2.7

## Usage
##### 1. Install dependencies

##### 2. Run code
For CIFAR-100 as an example(memory size/class 20)
```
bash train.sh 0   --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dual.yaml     --name DualVit     --data-path /your data-path/    --output-basedir /home/DAFormer/checkpoint  --memory-size 2000
```
For ImageNet-subset as an example(memory size/class 20)
```
bash train.sh 0    --options options/data/imagenet100_10-10.yaml options/data/imagenet100_order1.yaml options/model/imagenet_dual.yaml     --name DualVit     --data-path /your data-path/    --output-basedir /home/DAFormer/checkpoint  --memory-size 2000
```
