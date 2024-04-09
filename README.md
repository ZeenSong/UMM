# SSL_UMM
Official Code for "On the Generalization and Causal Explanation in Self-Supervised Learning"

## Install requirements
```
pip install -r requirements.txt
```

## Pretrain SSL models
```
accelerate launch pretrain.py --method simclr --model resnet18 --dataset cifar10
```

## Undoing Memorization Mechanism
```
accelerate launch train_umm.py --pretrain_path path -method simclr --model resnet18 --dataset cifar10
```