from itertools import chain
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


def get_backbone(cfg):
    if cfg.model == "resnet50":
        model = resnet50()
    else:
        model = resnet18()
    if 'imagenet' not in cfg.dataset: 
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
    if "cifar" in cfg.dataset:
        model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    return model


def get_MLP(in_dim, out_dim, hidden_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=False)
    )
    return mlp

class SwAV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
        self.prototypes = nn.Linear(cfg.out_dim, 100, bias=False)
    
    def get_representation(self, x1, x2):
        r1, r2 = self.back_bone(x1), self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head(r2)
        z1 = F.normalize(r1, p=2, dim=1)
        z2 = F.normalize(r2, p=2, dim=1)
        p1 = self.prototypes(z1)
        p2 = self.prototypes(z2)
        return p1, p2
    
    def forward(self, x1, x2):
        z1 = self.head(self.back_bone(x1))
        z2 = self.head(self.back_bone(x2))
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        p1 = self.prototypes(z1)
        p2 = self.prototypes(z2)
        return p1, p2

class SimCLR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.no_head = cfg.no_head
        if not cfg.no_head:
            if not cfg.linear_head:
                self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
            else:
                self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
    
    def get_representation(self, x1, x2):
        r1, r2 = self.back_bone(x1), self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head(r2)
        return z1, z2

    def forward(self, x1, x2):
        if self.no_head:
            z1 = self.back_bone(x1)
            z2 = self.back_bone(x2)
        else:
            z1 = self.head(self.back_bone(x1))
            z2 = self.head(self.back_bone(x2))
        return z1, z2
    

class MoCo(nn.Module):
    def __init__(self, cfg, K=4096):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.target = get_backbone(cfg)
        self.queue = F.normalize(torch.randn(K, cfg.out_dim).cuda()).detach()
        self.queue.requires_grad = False
        self.queue.ptr = 0
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
            self.head_target = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
            self.head_target = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
        for param in chain(self.target.parameters(), self.head_target.parameters()):
            param.requires_grad = False
    
    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.target.parameters(), self.back_bone.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_target.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
    
    def get_representation(self, x1, x2):
        r1 = self.back_bone(x1) 
        r2 = self.target(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head_target(r2)
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        return z1, z2
    
    def forward(self, x1, x2):
        z1 = self.head(self.back_bone(x1))
        z1 = F.normalize(z1, dim=1, p=2)
        
        with torch.no_grad():
            z2 = self.head_target(self.target(x2))
            z2 = F.normalize(z2, dim=1, p=2)
        return z1, z2

class BYOL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.target = get_backbone(cfg)
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
            self.head_target = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
            self.predictor = get_MLP(cfg.out_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
            self.head_target = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
            self.predictor = nn.Linear(cfg.out_dim, cfg.out_dim, bias=False)
        for param in chain(self.target.parameters(), self.head_target.parameters()):
            param.requires_grad = False
        self.byol_tau = cfg.byol_tau
        self.update_target(0)
    
    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.target.parameters(), self.back_bone.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_target.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
    
    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)
    
    def get_representation(self, x1, x2):
        r1 = self.back_bone(x1) 
        r2 = self.target(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, zt2 = self.predictor(self.head(r1)), self.head_target(r2)
        return z1, zt2
    
    def forward(self, x1, x2):
        z1 = self.predictor(self.head(self.back_bone(x1)))
        
        with torch.no_grad():
            zt2 = self.head_target(self.target(x2))
        return z1, zt2
        

class SimSiam(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        # if not cfg.linear_head:
        #     self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        #     self.predictor = get_MLP(cfg.out_dim, cfg.out_dim, cfg.hidden_dim)
        # else:
        #     self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
        #     self.predictor = nn.Linear(cfg.out_dim, cfg.out_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(cfg.rep_dim, cfg.rep_dim, bias=False),
            nn.BatchNorm1d(cfg.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rep_dim, cfg.rep_dim, bias=False),
            nn.BatchNorm1d(cfg.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False),
            nn.BatchNorm1d(cfg.out_dim, affine=False)
        )
        self.predictor = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.rep_dim, bias=False),
            nn.BatchNorm1d(cfg.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rep_dim, cfg.out_dim)
        )
    
    def get_representation(self, x1, x2):
        r1 = self.back_bone(x1) 
        r2 = self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        p1, z2 = self.predictor(self.head(r1)), self.head(r2)
        return p1, z2
    
    def forward(self, x1, x2):
        p1 = self.predictor(self.head(self.back_bone(x1)))
        z2 = self.head(self.back_bone(x2))
        return p1, z2

class VICReg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.no_head = cfg.no_head
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
    
    def get_representation(self, x1, x2):
        r1, r2 = self.back_bone(x1), self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head(r2)
        return z1, z2
    
    def forward(self, x1, x2):
        z1 = self.head(self.back_bone(x1))
        z2 = self.head(self.back_bone(x2))
        return z1, z2

class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.no_head = cfg.no_head
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
    
    def get_representation(self, x1, x2):
        r1, r2 = self.back_bone(x1), self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head(r2)
        return z1, z2
    
    def forward(self, x1, x2):
        z1 = self.head(self.back_bone(x1))
        z2 = self.head(self.back_bone(x2))
        return z1, z2

class Whitening2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, track_running_stats=True, eps=0):
        super(Whitening2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked),
            eye, 
            upper=False
            )
        
        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        decorrelated = nn.functional.conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)

    def extra_repr(self):
        return "features={}, eps={}, momentum={}".format(
            self.num_features, self.eps, self.momentum
        )
        
class WMSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.no_head = cfg.no_head
        if not cfg.linear_head:
            self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        else:
            self.head = nn.Linear(cfg.rep_dim, cfg.out_dim, bias=False)
    
    def get_representation(self, x1, x2):
        r1, r2 = self.back_bone(x1), self.back_bone(x2)
        return r1, r2

    def get_embedding(self, r1, r2):
        z1, z2 = self.head(r1), self.head(r2)
        return z1, z2
    
    def forward(self, x1, x2):
        z1 = self.head(self.back_bone(x1))
        z2 = self.head(self.back_bone(x2))
        return z1, z2

class MLPHead(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super(MLPHead, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.block(x)

class ReLIC(torch.nn.Module):

    def __init__(self,
                 cfg,
                 init_tau=1,
                 init_b=0):
        super(ReLIC, self).__init__()

        self.online_encoder = get_backbone(cfg)
        self.predictor = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)

        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        self.tau = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    @torch.inference_mode()
    def get_features(self, img):
        with torch.no_grad():
            return self.target_encoder[0](img)

    def forward(self, x1, x2):
        o1, o2 = self.online_encoder(x1), self.online_encoder(x2)
        with torch.no_grad():
            t1, t2 = self.target_encoder(x1), self.target_encoder(x2)
        t1, t2 = t1.detach(), t2.detach()
        return o1, o2, t1, t2
    
    @torch.inference_mode()
    def get_target_pred(self, x):
        with torch.no_grad():
            t = self.target_encoder(x)
        t = t.detach()
        return t
    
    def get_online_pred(self, x):
        return self.online_encoder(x)

    def update_params(self, gamma):
        with torch.no_grad():
            valid_types = [torch.float, torch.float16]
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1. - gamma)

            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1. - gamma)

    def copy_params(self):
        for o_param, t_param in self._get_params():
            t_param.data.copy_(o_param)

        for o_buffer, t_buffer in self._get_buffers():
            t_buffer.data.copy_(o_buffer)

    def save_encoder(self, path):
        torch.save(self.target_encoder[0].state_dict(), path)

    def _get_params(self):
        return zip(self.online_encoder.parameters(),
                   self.target_encoder.parameters())

    def _get_buffers(self):
        return zip(self.online_encoder.buffers(),
                   self.target_encoder.buffers())


class MEC(nn.Module):
    """
    Build a MEC model.
    """
    def __init__(self, cfg, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(MEC, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = get_backbone(cfg)

        # build a 2-layer predictor
        self.predictor = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)


        self.teacher = deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False
        lamda = 1 / (cfg.m * cfg.eps_d)
        self.scheduler = self.lamda_scheduler(8/lamda, 1/lamda, cfg.epochs, cfg.len, warmup_epochs=10)
        
            
    def lamda_scheduler(self, start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        z1 = self.predictor(self.encoder(x1)) # NxC
        z2 = self.predictor(self.encoder(x2)) # NxC

        with torch.no_grad():
            p1 = self.teacher(x1)
            p2 = self.teacher(x2)

        return z1, z2, p1.detach(), p2.detach()
    

class CorInfoMax(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.backbone = get_backbone(cfg)
        self.projector = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        self.normalized = cfg.normalize_on
        sizes = [512] + list(map(int, cfg.projector.split('-')))
        proj_output_dim = sizes[-1]
        self.R1 = cfg.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = cfg.R_ini*torch.eye(proj_output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu1 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_R2 = torch.zeros((proj_output_dim, proj_output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu2 = torch.zeros(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.la_R = cfg.la_R
        self.la_mu = cfg.la_mu
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = cfg.R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(proj_output_dim, dtype=torch.float64, device='cuda', requires_grad=False)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        # l-2 normalization of projector output 
        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2


class UMM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.back_bone = get_backbone(cfg)
        self.f_e = nn.Sequential(*list(self.back_bone.children())[:cfg.num_early])
        self.f_el = nn.Sequential(*list(self.back_bone.children())[cfg.num_early:])
        self.head = get_MLP(cfg.rep_dim, cfg.out_dim, cfg.hidden_dim)
        
    
    def get_Z_el(self, x):
        z_e = self.f_e(x)
        return z_e.detach()

    def get_Z(self, x):
        z = self.head(self.back_bone(x))
        return z
    
    def forward(self, x1, x2):
        z_e1, z_e2 = self.f_e(x1), self.f_e(x2)
        
        z1 = self.head(self.f_el(z_e1))
        z2 = self.head(self.f_el(z_e2))
        
        return z1, z2, z_e1, z_e2