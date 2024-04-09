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
from ssl_losses import *
from models import *
from betty.problems import ImplicitProblem
from betty.engine import Engine
from betty.configs import Config, EngineConfig

trainers_dict = {
    "simclr": simclr_loss,
    "swav": swav_loss,
    "moco": moco_loss,
    "byol": byol_loss,
    "simsiam": simsiam_loss,
    "barlow": barlow_loss,
    "vicreg": vicreg_loss,
    "wmse": wmse_loss,
    "relic": relic_loss,
    "mec": mec_loss,
    "corinfomax": corinfomax_loss
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


class UMM(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.model = model(cfg)
        self.back_bone = self.model.back_bone
        self.back_bone.fc = nn.Identity()
        self.back_bone.maxpool = nn.Identity()
        state_dict = torch.load(cfg.pretrain_path)
        self.back_bone.load_state_dict(state_dict)
        self.f_e = nn.Sequential(
            *list(self.back_bone.children())[:cfg.num_early])
        self.f_el = nn.Sequential(
            *list(self.back_bone.children())[cfg.num_early:])
        self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        self.mcr = MaximalCodingRateReduction()
        self.alpha = cfg.alpha
        self.beta = cfg.beta

    def forward(self, x1, x2):
        with torch.no_grad():
            z_e1, z_e2 = self.f_e(x1), self.f_e(x2)
        z_e1.requires_grad = True
        z_e2.requires_grad = True
        r1, r2 = self.f_el(z_e1).squeeze(), self.f_el(z_e2).squeeze()
        z1 = self.head(r1)
        z2 = self.head(r2)

        return z1, z2, z_e1, z_e2


def calc_prob(loss, z_e1, z_e2):
    Jacobian_1 = torch.autograd.grad(loss, z_e1, retain_graph=True)[
        0].view(z_e1.shape[0], -1)
    Jacobian_2 = torch.autograd.grad(loss, z_e2, retain_graph=True)[
        0].view(z_e1.shape[0], -1)
    N = z_e1.shape[0]
    prob = torch.zeros(2*N).cuda()
    for i in range(N):
        prob[i] = 1 / (torch.sqrt(Jacobian_1[i].pow_(2).sum())*(2*N))
        prob[i+N] = 1 / (torch.sqrt(Jacobian_2[i].pow_(2).sum())*(2*N))
    prob = prob / prob.sum()
    return prob


def js_div(prob):
    prob_1 = torch.ones_like(prob) / prob.shape[0]
    M = 0.5 * (prob + prob_1)
    loss = 0.5 * (F.kl_div(prob, M)+F.kl_div(prob_1, M))
    return loss


class Inner(ImplicitProblem):
    def __init__(self, name, config, trainer, args, accelerator, module=None, optimizer=None, scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer,
                         scheduler, train_data_loader, extra_config)
        self.trainer = trainer
        self.args = args
        self.accelerator = accelerator

    def training_step(self, batch):
        x1, x2 = batch
        z1, z2, z_e1, z_e2 = self.module(x1.cuda(), x2.cuda())
        ssl_loss = self.trainer(z1, z2, self.args, self.accelerator)
        prob = calc_prob(ssl_loss, z_e1, z_e2)
        loss_2 = js_div(prob)
        total_loss = ssl_loss + self.args.beta * loss_2
        return total_loss

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())


class Outer(ImplicitProblem):
    def __init__(self, name, config, trainer, args, accelerator, module=None, optimizer=None, scheduler=None, train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer,
                         scheduler, train_data_loader, extra_config)
        self.trainer = trainer
        self.args = args
        self.accelerator = accelerator

    def training_step(self, batch):
        x1, x2 = batch
        z1, z2, z_e1, z_e2 = self.module(x1.cuda(), x2.cuda())
        loss = self.trainer(z1, z2, self.args, self.accelerator)
        Z = torch.cat([z1, z2], dim=0)
        N = z1.shape[0]
        logits = torch.cat([torch.arange(N), torch.arange(N)], dim=0)
        loss_crr = self.module.mcr(Z, logits)
        total_loss = loss + loss_crr
        return total_loss


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


class UMMEngine(Engine):
    def __init__(self, problems, model, clf_loader, test_loader, cfg, accelerator: Accelerator, config=None, dependencies=None, env=None):
        super().__init__(problems, config, dependencies, env)
        self.model = model
        self.clf_loader = clf_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.accelerator = accelerator

    def validation(self):
        # prepare classifier
        lr_start, lr_end = 1e-2, 1e-6
        gamma = (lr_end / lr_start) ** (1 / 500)
        classifier = nn.Linear(cfg.rep_dim, 1000)
        classifier.train()
        classifier.cuda()
        optimizer = Adam(classifier.parameters(),
                         lr=lr_start, weight_decay=1e-6)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        if self.accelerator.num_processes > 1:
            target = self.model.module
        else:
            target = self.model
        classifier, optimizer = self.accelerator.prepare(classifier, optimizer)
        x_train, y_train = collect_features(
            target, self.clf_loader, self.accelerator.device)
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
                self.accelerator.backward(loss)
                optimizer.step()
            scheduler.step()
        # test classifier in main node
        classifier.eval()
        if self.accelerator.process_index == 0:
            correct = 0
            total = 0
            pbar = tqdm(self.test_loader)
            pbar.set_description("evaluate classifier")
            for i, (x, y) in enumerate(pbar):
                x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    r = target.back_bone(x)
                    y_pred = classifier(r)
                predictions = y_pred.argmax(dim=-1)
                accurate_preds = predictions == y
                correct += accurate_preds.long().sum().item()
                total += x.shape[0]
            acc = correct / total


def get_dataset(dataset):
    if dataset == "stl10":
        train_set = UnlabelSTL10(
            root='data/stl10/unlabel', transform=StlPairTransform())
        clf_set = ImageFolder(root='data/stl10/train', transform=StlPairTransform(
            train_transform=False, pair_transform=False))
        test_set = ImageFolder(
            root='data/stl10/test', transform=StlPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "cifar10":
        train_set = UnlabelCIFAR10(
            root='data', train=True, transform=CIFARPairTransform())
        clf_set = CIFAR10(root='data', train=True, transform=CIFARPairTransform(
            train_transform=False, pair_transform=False))
        test_set = CIFAR10(root='data', train=False, transform=CIFARPairTransform(
            train_transform=False, pair_transform=False))
    elif dataset == "tinyim":
        train_set = UnlabelSTL10(
            root='data/tiny-imagenet-200/train/', transform=TinyImPairTransform())
        clf_set = ImageFolder(root='data/tiny-imagenet-200/train/',
                              transform=TinyImPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/tiny-imagenet-200/val/',
                               transform=TinyImPairTransform(train_transform=False, pair_transform=False))
    elif dataset == "cifar100":
        train_set = UnlabelCIFAR100(
            root='data', train=True, transform=CIFARPairTransform())
        clf_set = CIFAR100(root='data', train=True, transform=CIFARPairTransform(
            train_transform=False, pair_transform=False))
        test_set = CIFAR100(root='data', train=False, transform=CIFARPairTransform(
            train_transform=False, pair_transform=False))
    elif dataset == "imagenet":
        train_set = UnlabelSTL10(
            root='data/ImageNet/train/', transform=ImageNetPairTransform())
        clf_set = ImageFolder(root='data/ImageNet/clf/', transform=ImageNetPairTransform(
            train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/ImageNet/val/', transform=ImageNetPairTransform(
            train_transform=False, pair_transform=False))
    elif dataset == "imagenet100":
        train_set = UnlabelSTL10(
            root='data/ImageNet100/train/', transform=ImageNetPairTransform())
        clf_set = ImageFolder(root='data/ImageNet100/train/',
                              transform=ImageNetPairTransform(train_transform=False, pair_transform=False))
        test_set = ImageFolder(root='data/ImageNet100/val/', transform=ImageNetPairTransform(
            train_transform=False, pair_transform=False))
    else:
        raise NotImplementedError("The dataset is not supported")
    return train_set, clf_set, test_set


def run(cfg):
    # initialize project tensorboard logging
    accelerator = Accelerator(log_with='wandb')
    if accelerator.process_index == 0:
        dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + \
            f"{cfg.method}_{cfg.dataset}"
        os.mkdir("logs/"+dir_name)
        accelerator.init_trackers(project_name="SSL_UMM", config=cfg.__dict__)
        accelerator.trackers[0].run.name = f'{cfg.method}_{cfg.dataset}'

    # prepare data
    train_set, clf_set, test_set = get_dataset(cfg.dataset)
    train_loader = DataLoader(
        train_set, batch_size=cfg.train_bs, num_workers=4, pin_memory=False, shuffle=True)
    clf_loader = DataLoader(clf_set, batch_size=cfg.eval_bs,
                            num_workers=4, pin_memory=False, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=cfg.eval_bs, num_workers=4, pin_memory=False, shuffle=True)

    # prepare model and optimizer
    model = model_dict[cfg.method]
    model = UMM(cfg, model)
    outer_optim = Adam(model.f_el.parameters(), lr=cfg.lr,
                       weight_decay=cfg.weight_decay)
    inner_optim = Adam(model.f_el.parameters(), lr=cfg.lr,
                       weight_decay=cfg.weight_decay)

    # init with accelerator
    model, train_loader, clf_loader, test_loader = accelerator.prepare(
        model, train_loader, clf_loader, test_loader)

    # create trainer and evaluator
    trainer = trainers_dict[cfg.method]

    outer_config = Config(type="darts", retain_graph=True)
    inner_config = Config(type="darts", unroll_steps=1)
    engine_config = EngineConfig(train_iters=cfg.train_iters)

    outer = Outer(name="outer", accelerator=accelerator, module=model, config=outer_config,
                  optimizer=outer_optim, train_data_loader=train_loader, trainer=trainer, args=cfg)
    inner = Inner(name="inner", accelerator=accelerator, module=model, config=inner_config,
                  optimizer=inner_optim, train_data_loader=train_loader, trainer=trainer, args=cfg)

    # start training
    problems = [outer, inner]
    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
    dependencies = {"u2l": u2l, "l2u": l2u}
    engine = UMMEngine(
        config=engine_config, problems=problems, dependencies=dependencies, 
        model=model, clf_loader=clf_loader, 
        test_loader=clf_loader, cfg=cfg, accelerator=accelerator
    )

    engine.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_freq', type=int)
    parser.add_argument('--pretrain_path',type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--method', type=str)
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
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--tau', type=float)
    parser.add_argument('--warm_up_epoch', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--byol_tau', type=float)
    parser.add_argument('--mid_dim', type=int)
    parser.add_argument('--num_early', type=int)
    parser.add_argument('--train_iters', type=int)
    parser.add_argument('--beta', type=float)

    cfg = parser.parse_args()
    run(cfg)
