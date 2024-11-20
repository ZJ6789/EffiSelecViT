## EffiSelecViT

## Environment


- python 3.10
- pytorch >= 1.7
- torchvision >= 0.8
- timm 0.5.4

## Data prepare
Please refer to the DeiT repository to prepare the standard ImageNet dataset, then link the ImageNet dataset under the datafolder:


```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```    
`References:` 

`H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. J´egou,
Training data-efficient image transformers & distillation through attention, in: International conference on machine learning, PMLR, 2021, pp.10347–10357.`

## Getting Started
Our approach consists of a search and fine-tuning phase:

- Search

  Performing an importance search on components of the pre-trained model and select the important components.

- Fine-tuning

  Based on the searched importance proxy scores, following budgets at different scales, we prune the unimportant components, and then perform fine-tuning on the pruned model to recover accuracy.



Here, we take the pruning DeiT-B model as an example and provide the corresponding program to execute the command line.

- Search

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env search.py --data-path /path/to/dataset --model deit_base_patch16_224 --begin_search --search_head --head_w 1e-2 --search_mlp --mlp_w 1e-4 --pretrained_path /path/to/original/pre-trained/checkpoint --output_dir /path/to/save
```

- Fine-tuning

In this phase, the FLOPs can also be displayed when running the command.
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env finetune.py --data-path /path/to/dataset --retrain --search_checkpoint /path/to/search_checkpoint --prune_head --head_prune_ratio 0.3 --prune_mlp --mlp_prune_ratio 0.5 --checkpoint_path /path/to/original/pre-trained/checkpoint --output_dir /path/output
```


