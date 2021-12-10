import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, StepLR


def make_optimizer(cfg, model):
    optimizer = None
    if cfg.solver.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.solver.learning_rate,
                                     weight_decay=cfg.solver.weight_decay)
    elif cfg.solver.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.solver.learning_rate,
                                      weight_decay=cfg.solver.weight_decay)
    return optimizer


def make_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.solver.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1)
    elif cfg.solver.scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif cfg.solver.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=00)
    elif cfg.solver.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=6000, gamma=0.1)
    elif cfg.solver.scheduler == 'ConstantLR':
        scheduler = StepLR(optimizer, step_size=100, gamma=1)
    return scheduler