import torch
import torch.nn as nn
import torch.autograd

import torch.nn.functional as F

from accelerate import Accelerator
from models import *


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        # c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()

def contrastive_loss(x0, x1, tau, norm, accelerator: Accelerator):
    # https://github.com/google-research/simclr/blob/master/objective.py
    world_size = accelerator.num_processes
    bsize = x0.shape[0]
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    x0_large = accelerator.gather(x0)
    x1_large = accelerator.gather(x1)
    cur_rank = accelerator.process_index
    target = torch.arange(cur_rank*bsize, (cur_rank+1)*bsize).cuda()
    eye_mask = F.one_hot(target, num_classes=world_size*bsize).cuda() * 1e9

    logits00 = x0 @ x0_large.t() / tau - eye_mask
    logits11 = x1 @ x1_large.t() / tau - eye_mask
    logits01 = x0 @ x1_large.t() / tau
    logits10 = x1 @ x0_large.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2

def simclr_loss(z1, z2, cfg, accelerator: Accelerator):
    loss = contrastive_loss(z1, z2, cfg.tau, cfg.norm, accelerator)
    return loss

def swav_loss(p1, p2, epsilon=0.05, n_iters=3, temperature=0.1):
    q1 = distributed_sinkhorn(torch.exp(p1 / epsilon).t(), n_iters)
    q2 = distributed_sinkhorn(torch.exp(p2 / epsilon).t(), n_iters)
    
    p1 = F.softmax(p1 / temperature, dim=1)
    p2 = F.softmax(p2 / temperature, dim=1)

    loss1 = -torch.mean(torch.sum(q1 * torch.log(p2), dim=1))
    loss2 = -torch.mean(torch.sum(q2 * torch.log(p1), dim=1))
    loss = loss1+loss2
    return loss

def swav_loss(batch, model):
    model.train()
    x1, x2 = batch
    x1, x2 = x1.cuda(), x2.cuda()
    p1, p2 = model(x1, x2)
    loss = swav_loss(p1, p2)
    return loss

def moco_loss(z1, z2, queue, T=0.2):
    l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
    l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1).div(T)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss

def moco_loss(batch, model, accelerator: Accelerator, K=4096, momentum=0.999):
    model.train()
    x1, x2 = batch
    x1, x2 = x1.cuda(), x2.cuda()
    if accelerator.num_processes > 1:
        target = model.module
    else:
        target = model
    z1, z2 = model(x1, x2)
    loss = moco_loss(z1, z2, target.queue)

        
    target.update_target(momentum)
    keys = accelerator.gather(z1)
    # not excess the total length
    if target.queue.ptr+keys.shape[0] < K:
        target.queue[target.queue.ptr:target.queue.ptr+keys.shape[0]] = keys
    # circle queue
    else:
        target.queue[target.queue.ptr:K] = keys[:K-target.queue.ptr]
        target.queue[:keys.shape[0] - K + target.queue.ptr] = keys[K-target.queue.ptr:]
    target.queue.ptr = (target.queue.ptr+keys.shape[0]) % K
    
    return loss
        
def byol_loss(batch, model, cfg, accelerator: Accelerator):
    x1, x2 = batch
    x1, x2 = x1.cuda(), x2.cuda()
    if accelerator.num_processes > 1:
        target = model.module
    else:
        target = model
    z1, zt2 = model(x1, x2)
    z2, zt1 = model(x2, x1)
    loss1 = norm_mse_loss(z1, zt2)
    loss2 = norm_mse_loss(z2, zt1)
    loss = (loss1+loss2).mul(0.5)

    target.step(cfg.epoch / cfg.max_epochs)
    
    return loss

def simsiam_loss(batch, model):
    model.train()
    x1, x2 = batch
    x1, x2 = x1.cuda(), x2.cuda()
    p1, z2 = model(x1, x2)
    p2, z1 = model(x2, x1)
    loss1 = F.cosine_similarity(p1, z2.detach(), dim=-1).mean().mul(-1)
    loss2 = F.cosine_similarity(p2, z1.detach(), dim=-1).mean().mul(-1)
    loss = (loss1+loss2).mul(0.5)
    
    return loss

def barlow_loss(z1, z2):
    bs = z1.shape[0]
    z1 = (z1 - z1.mean(0)) / z1.std(0) # NxD
    z2 = (z2 - z2.mean(0)) / z2.std(0) # NxD
    lambd = 1 / z1.shape[1]
    c = torch.matmul(z1.T, z2) / bs

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss


def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    repr_loss = F.mse_loss(z1, z2)
    
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    
    std_loss = torch.mean(F.relu(1-std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2
    
    bs = z1.shape[0]
    num_d = z1.shape[1]
    cov_z1 = (z1.T @ z1) / bs
    cov_z2 = (z2.T @ z2) / bs
    
    cov_loss = off_diagonal(cov_z1).pow_(2).sum().div(num_d) + off_diagonal(cov_z2).pow_(2).sum().div(num_d)
    
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss


def wmse_loss(z1, z2, w_iter=1, w_size=128):
    bs = z1.shape[0]
    h = [z1,z2]
    whitening = Whitening2d(z1.shape[1])
    for _ in range(w_iter):
        z = torch.empty_like(h)
        perm = torch.randperm(bs).view(-1, w_size)
        for idx in perm:
            for i in range(len(h)):
                z[idx + i * bs] = whitening(h[idx + i * bs])
        for i in range(len(h) - 1):
            for j in range(i + 1, len(h)):
                x0 = z[i * bs : (i + 1) * bs]
                x1 = z[j * bs : (j + 1) * bs]
                loss += norm_mse_loss(x0, x1)
    loss /= w_iter * 2

    return loss

def relic_loss(z1, z2, model, max_tau=5.0):
    x, x_prime = z1, z2
    n = x.size(0)
    logits = torch.mm(x, x_prime.t()) * model.tau.exp().clamp(0, max_tau) + model.b
    
    labels = torch.arange(n).to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    # KL divergence loss
    p1 = torch.nn.functional.log_softmax(logits, dim=1)
    p2 = torch.nn.functional.softmax(logits, dim=0).t()
    invariance_loss = torch.nn.functional.kl_div(p1, p2, reduction="batchmean")

    loss = loss + model.alpha * invariance_loss

    # return invariance_loss for debug
    return loss

def mec_loss(p, z, lamda_inv, accelerator: Accelerator, order=4):

    p = accelerator.gather(p)
    z = accelerator.gather(z)

    p = F.normalize(p)
    z = F.normalize(z)

    c = p @ z.T

    c = c / lamda_inv 

    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else: 
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace

def corinfomax_loss(z1,z2, model):
    
    la_R = model.la_R
    la_mu = model.la_mu

    N, D = z1.size()

    # mean estimation
    mu_update1 = torch.mean(z1, 0)
    mu_update2 = torch.mean(z2, 0)
    model.new_mu1 = la_mu*(model.mu1) + (1-la_mu)*(mu_update1)
    model.new_mu2 = la_mu*(model.mu2) + (1-la_mu)*(mu_update2)

    # covariance matrix estimation
    z1_hat =  z1 - model.new_mu1
    z2_hat =  z2 - model.new_mu2
    R1_update = (z1_hat.T @ z1_hat) / N
    R2_update = (z2_hat.T @ z2_hat) / N
    model.new_R1 = la_R*(model.R1) + (1-la_R)*(R1_update)
    model.new_R2 = la_R*(model.R2) + (1-la_R)*(R2_update)

    # loss calculation 
    cov_loss = - (torch.logdet(model.new_R1 + model.R_eps) + torch.logdet(model.new_R2 + model.R_eps)) / D

    # This is required because new_R updated with backward.
    model.R1 = model.new_R1.detach()
    model.mu1 = model.new_mu1.detach()
    model.R2 = model.new_R2.detach()
    model.mu2 = model.new_mu2.detach()

    return cov_loss
