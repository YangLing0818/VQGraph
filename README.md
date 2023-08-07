# VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs
<a href="https://arxiv.org/abs/2308.02117"><img src="https://img.shields.io/badge/arXiv-2308.02117-brown.svg" height=22.5></a>

This repository is the official implementation of [VQGraph](). VQGraph is the state-of-the-art (SOTA) GNN-to-MLP distillation method.

[VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs]().

Authors: Ling Yang, Ye Tian, Minkai Xu, Zhongyi Liu, Shenda Hong, Wei Qu, Wentao Zhang, Bin Cui, Muhan Zhang, Jure Leskovec



## Overview
---
![Alt text](image.png)


## Updates
---

- [04/08/2023] Code released

## TODO
---

- [ ] Release the code of graph tokenizer training
- [ ] Release the code of parallel training

## Requirements 
---
* torch >= 1.7.0
* ogb >= 1.3.3
* dgl >= 0.6.1
* networkx >= 2.5.1
* googledrivedownloader >= 0.4
* category_encoders >= 2.3.0
* einops >= 0.6.0

## Datasets
---
Please download the datasets, and put them under `data/` (see below for instructions on organizing the datasets).

- *CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://www.dropbox.com/sh/fchrckrpf99gho2/AABZwMOeOnuiCxBjqYd46Qz3a?dl=0). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

- *OGB data* (`ogbn-arxiv` and `ogbn-products`): Datasets will be automatically downloaded when running the `load_data` function in `dataloader.py`. Please refer to the OGB official website for more details.

## Training and Evaluation
---

**Teacher Model**: Our pretrained codebook embeddings, teacher soft assignments and teacher soft labels for some datasets have been uploaded to [here](https://www.dropbox.com/scl/fo/9yss598aln21gzdiwix61/h?dl=0&rlkey=oscheo12z9md8uah7eakq62yj). Please download and put them under `outputs/transductive/{dataset}/GCN/` for GNN-MLP distillation.


**GNN-to-MLP Distillation**: To quickly reproduce our VQGraph, you can run `train_student.py` by specifying the experiment setting, including teacher model, student model, output path of the teacher model and dataset like the following example command: 

```
python train_student.py --exp_setting tran --teacher GCN --student MLP --dataset citeseer --out_t_path outputs --seed 0 --max_epoch 500 --patience 50 --device 0
```

## Citation
---
If you found the codes are useful, please cite our paper
```
@article{yang2023vqgraph,
  title={VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs},
  author={Yang, Ling and Tian, Ye and Xu, Minkai and Liu, Zhongyi and Hong, Shenda and Qu, Wei and Zhang, Wentao and Cui, Bin and Zhang, Muhan and Leskovec, Jure},
  journal={arXiv preprint arXiv:2308.02117},
  year={2023}
}
```