import logging
import math
import os
import time
import torch.nn as nn
import argparse
from tqdm import tqdm
import shutil
import pdb
import torch

import utils
from utils import setup_logger
from model import Model
from train import Trainer
from dataset import make_dataloader
from optim import make_optimizer, make_scheduler
from loss import MatrixSoftmaxCELoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='config.json',
                        help='the path to the training config')
    parser.add_argument('-t', '--test', action='store_true',
                        default=False, help='Model test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)


def train(args):
    cfg = utils.process_cfg(args.config_file)
    output_dir = os.path.join(cfg.exp_base, cfg.exp_name, str(time.time()))
    cfg.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, cfg.output_dir)
    setup_logger(output_dir)
    logger = logging.getLogger()
    logger.info('Train with config:\n{}'.format(cfg))

    model = Model(cfg).to(cfg.device)
    logger.info("model architecture:")
    logger.info(model)

    train_dl = make_dataloader(cfg, 'train')
    val_dl = make_dataloader(cfg, 'validation')

    optimizer = make_optimizer(cfg, model)

    scheduler = make_scheduler(cfg, optimizer)

    loss_func = MatrixSoftmaxCELoss().to(cfg.device)

    trainer = Trainer(cfg, model, train_dl, val_dl, optimizer, scheduler, loss_func)

    trainer.train()


def test(args):
    cfg = utils.process_cfg(args.config_file)
    output_dir = os.path.join(cfg.exp_base, cfg.exp_name, str(time.time()))
    cfg.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copy(args.config_file, cfg.output_dir)
    setup_logger(output_dir)
    logger = logging.getLogger()
    logger.info('Test with config:\n{}'.format(cfg))

    model, preprocess = load(cfg.device)
    logger.info("model architecture:")
    logger.info(model)

    train_dl = make_dataloader(cfg, 'train', preprocess)
    val_dl = make_dataloader(cfg, 'validation', preprocess)

    optimizer = make_optimizer(cfg, model)

    scheduler = make_scheduler(cfg, optimizer)

    loss_func = MatrixSoftmaxCELoss()

    trainer = Trainer(cfg, model, train_dl, val_dl, optimizer, scheduler, loss_func)

    top1_acc, top5_acc, top10_acc = trainer.evaluate()
    print(top1_acc, top5_acc, top10_acc)


if __name__ == '__main__':
    main()