import torch
from torch.optim.lr_scheduler import *
from torch.optim import *

class MyScheduler(object):
    def __init__(self, warm_up_epoch: int, step_epochs: list, max_epochs: int, optimizer: Optimizer, step_scale: float, init_lr: float):
        self.warm_up_epoch = warm_up_epoch
        self.step_epochs = step_epochs
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.step_scale = step_scale
        self.cos_scheduler = CosineAnnealingLR(optimizer, self.max_epochs - self.warm_up_epoch, eta_min=1e-5)
        self.step_scheduler = MultiStepLR(optimizer, milestones=self.step_epochs, gamma=self.step_scale)
        self.init_lr = init_lr
    
    def step(self, cur_epoch:int):
        # warm up
        if cur_epoch < self.warm_up_epoch:
            lr_scale = (cur_epoch + 1) / self.warm_up_epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr * lr_scale
        else:
            self.cos_scheduler.step()
        self.step_scheduler.step()