import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, StepLR


def make_optimizer(cfg, model):
    optimizer = None
    if cfg.solver.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.solver.learning_rate,
                                     weight_decay=cfg.solver.weight_decay)
    elif cfg.solver.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.solver.learning_rate,
                                      weight_decay=cfg.solver.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.solver.learning_rate,
        #                               weight_decay=cfg.solver.weight_decay)
    return optimizer


def make_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.solver.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.solver.T_0, T_mult=cfg.solver.T_mult)
    elif cfg.solver.scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=cfg.solver.gamma)
    elif cfg.solver.schedule == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.solver.T_max)
    elif cfg.solver.schedule == 'ConstantLR':
        scheduler = StepLR(optimizer, step_size=cfg.solver.num_epochs, gamma=1)
    return scheduler