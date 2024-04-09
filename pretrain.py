import os
from argparse import ArgumentParser
import json
import datetime

import torch
import torch.autograd
from torch.optim import SGD, Adam, AdamW, LBFGS
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder, CIFAR100, CIFAR10
from torchvision.models import resnet18, resnet50
from MCR_class import MaximalCodingRateReduction

from accelerate import Accelerator
from tqdm import tqdm

from dataset import StlPairTransform, CIFARPairTransform, UnlabelCIFAR100, UnlabelSTL10, ImageNetPairTransform, UnlabelCIFAR10, TinyImPairTransform
from trainers import *
from models import *

trainers_dict = {
    "simclr": simclr_trainer,
    "swav": swav_trainer,
    "moco": moco_trainer,
    "byol": byol_trainer,
    "simsiam": simsiam_trainer,
    "barlow": barlow_trainer,
    "vicreg": vicreg_trainer,
    "wmse": wmse_trainer,
    "relic": relic_trainer,
    "mec": mec_trainer,
    "corinfomax": corinfomax_trainer
}

model_dict = {
    "simclr": SimCLR,
    "moco": MoCo,
    "swav": SwAV,
    "byol": BYOL,
    "simsiam": SimSiam,
    "barlow": BarlowTwins,
    "vicreg": VICReg,
    "wmse": WMSE,
    "relic": ReLIC,
    "mec": MEC,
    "corinfomax": CorInfoMax
}

def collect_features(model,
                     dataloader,
                     device):

    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        pbar = tqdm(dataloader)
        for i, (x, y) in enumerate(pbar):
            z = model.back_bone(x.to(device))
            features.append(z.to(device).detach())
            labels.append(y.to(device).detach())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def evaluate(model, clf_loader, test_loader, cfg, epoch, accelerator: Accelerator):
    # prepare classifier
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / 500)
    classifier = nn.Linear(cfg.rep_dim, 1000)
    classifier.train()
    classifier.cuda()
    optimizer = Adam(classifier.parameters(), lr=lr_start, weight_decay=1e-6)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    if accelerator.num_processes > 1:
        target = model.module
    else:
        target = model
    classifier, optimizer = accelerator.prepare(classifier, optimizer)
    x_train, y_train = collect_features(target, clf_loader, accelerator.device)
    # train classifier distributed
    pbar = tqdm(range(1000))
    for ep in pbar:
        perm = torch.randperm(len(x_train)).view(-1, 512)
        total = 0
        correct = 0
        for index in perm:
            r = x_train[index]
            optimizer.zero_grad()
            y_pred = classifier(r)
            loss = F.cross_entropy(y_pred, y_train[index])
            predictions = y_pred.argmax(dim=-1)
            accurate_preds = predictions == y_train[index]
            correct += accurate_preds.long().sum().item()
            total += r.shape[0]
            acc = correct / total
            pbar.set_description(f"ACC: {acc*100} %")
            accelerator.backward(loss)
            optimizer.step()
        scheduler.step()
    accelerator.log({"train acc": acc}, step=epoch)
    # test classifier in main node
    classifier.eval()
    if accelerator.process_index == 0:
        correct = 0
        total = 0
        pbar = tqdm(test_loader)
        pbar.set_description("evaluate classifier")
        for i, (x,y) in enumerate(pbar):
            x,y = x.cuda(), y.cuda()
            with torch.no_grad():
                r = target.back_bone(x)
                y_pred = classifier(r)
            predictions = y_pred.argmax(dim=-1)
            accurate_preds = predictions == y
            correct += accurate_preds.long().sum().item()
            total += x.shape[0]
        acc = correct / total
        print(correct, total, acc)
        accelerator.log({"linear eval acc": acc}, step=epoch)
                
                

def get_dataset(dataset):
    if dataset == "stl10":
        train_set = UnlabelSTL10(root='data/stl10/unlabel', transform=StlPairTransform())
        clf_set = ImageFolder(root='data/stl10/train', transform=StlPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/stl10/test', transform=StlPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "cifar10":
        train_set = UnlabelCIFAR10(root='data', train=True, transform=CIFARPairTransform())
        clf_set = CIFAR10(root='data', train=True, transform=CIFARPairTransform(train_transform=False, pair_transform=False))
        test_set = CIFAR10(root='data', train=False, transform=CIFARPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "tinyim":
        train_set = UnlabelSTL10(root='data/tiny-imagenet-200/train/', transform=TinyImPairTransform())
        clf_set = ImageFolder(root='data/tiny-imagenet-200/train/', transform=TinyImPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/tiny-imagenet-200/val/', transform=TinyImPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "cifar100":
        train_set = UnlabelCIFAR100(root='data', train=True, transform=CIFARPairTransform())
        clf_set = CIFAR100(root='data', train=True, transform=CIFARPairTransform(train_transform=False, pair_transform=False))
        test_set = CIFAR100(root='data', train=False, transform=CIFARPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "imagenet":
        train_set = UnlabelSTL10(root='data/ImageNet/train/', transform=ImageNetPairTransform())
        clf_set = ImageFolder(root='data/ImageNet/clf/', transform=ImageNetPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/ImageNet/val/', transform=ImageNetPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "imagenet100":
        train_set = UnlabelSTL10(root='data/ImageNet100/train/', transform=ImageNetPairTransform())
        clf_set = ImageFolder(root='data/ImageNet100/train/', transform=ImageNetPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/ImageNet100/val/', transform=ImageNetPairTransform(train_transform=False, pair_transform=False))
    else:
        raise NotImplementedError("The dataset is not supported")
    return train_set, clf_set, test_set

def run(cfg):
    # initialize project tensorboard logging
    accelerator = Accelerator(log_with='wandb')
    if accelerator.process_index == 0:
        dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"{cfg.method}_{cfg.dataset}"
        os.mkdir("logs/"+dir_name)
        accelerator.init_trackers(project_name="SSL_UMM", config=cfg.__dict__)
        accelerator.trackers[0].run.name = f'{cfg.method}_{cfg.dataset}'
    
    # prepare data
    train_set, clf_set, test_set = get_dataset(cfg.dataset)
    train_loader = DataLoader(train_set, batch_size=cfg.train_bs, num_workers=4, pin_memory=False, shuffle=True)
    clf_loader = DataLoader(clf_set, batch_size=cfg.eval_bs, num_workers=4, pin_memory=False, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=cfg.eval_bs, num_workers=4, pin_memory=False, shuffle=True)
    
    # prepare model and optimizer
    model = model_dict[cfg.method](cfg)
    optimizer = Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # init with accelerator
    model, optimizer, train_loader, clf_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, clf_loader, test_loader)
    
    # create trainer and evaluator
    trainer = trainers_dict[cfg.method]
    
    # start training
    total = len(train_loader)
    for epoch in range(cfg.max_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f"pretrain: epoch {epoch}/{cfg.max_epochs}")
        for i, batch in enumerate(pbar):
            cfg.epoch = epoch
            loss = trainer(batch, model, optimizer, cfg, accelerator)
            if accelerator.process_index == 0:
                accelerator.log({"training loss": loss.item()})
        model.eval()
        if epoch % cfg.eval_freq == 0:
            evaluate(model, clf_loader, test_loader, cfg, epoch, accelerator)
        if (epoch+1) % cfg.save_freq == 0:
            if accelerator.num_processes > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            if accelerator.process_index == 0:
                torch.save(state_dict, f'./logs/{dir_name}/{cfg.method}_{epoch+1}.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--eval_freq', type=int)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--t0', type=int)
    parser.add_argument('--t_mult', type=int)
    parser.add_argument('--train_bs', type=int)
    parser.add_argument('--eval_bs', type=int)
    parser.add_argument('--no_head', action='store_true')
    parser.add_argument('--rep_dim', type=int)
    parser.add_argument('--hidden_dim',type=int)
    parser.add_argument('--out_dim',type=int)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--tau', type=float)
    parser.add_argument('--warm_up_epoch', type=int)
    parser.add_argument('--step_scale', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--byol_tau', type=float)
    parser.add_argument('--mid_dim', type=int)
    
    cfg = parser.parse_args()
    run(cfg)