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
import matplotlib.pyplot as plt
from skimage.io import imread
import clip

import utils
from utils import setup_logger
from train import Trainer
from dataset import make_dataloader
from optim import make_optimizer, make_scheduler
from loss import InfoNCELoss

from utils import AverageMeter


def load(device):
    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    return clip_model, preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='clip_fine_tune/config.json',
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

    model, preprocess = load(cfg.device)
    logger.info("model architecture:")
    logger.info(model)

    train_dl = make_dataloader(cfg, 'train', preprocess)
    val_dl = make_dataloader(cfg, 'validation', preprocess)

    optimizer = make_optimizer(cfg, model)

    scheduler = make_scheduler(cfg, optimizer)

    loss_func = InfoNCELoss()

    trainer = Trainer(cfg, model, train_dl, val_dl, optimizer, scheduler, loss_func)

    trainer.train()

#
# def test(args):
#     cfg = utils.process_cfg(args.config_file)
#     test_dl = make_dataloader(cfg, 'test')
#     model = build_model(cfg)
#     model.load_state_dict(torch.load(cfg.test.model_path))
#     model = model.to(cfg.device)
#     loss_func = nn.CrossEntropyLoss()
#     device = cfg.device
#     model.eval()
#     loss_avg = AverageMeter()
#     stats = torch.zeros((2, 2))  # [[tp, fp], [fn, tn]]
#     fp_paths, fn_paths = [], []
#     for data, label, paths in tqdm(test_dl):
#         data, label = data.to(device), label.to(device)
#         probs = model(data)
#         loss = loss_func(probs, label)
#         preds = torch.argmax(probs, dim=1)
#         preds = preds.detach().cpu()
#         label = label.detach().cpu()
#         confuse_matrix = \
#             torch.matmul(torch.stack([preds, 1 - preds], dim=0), torch.stack([label, 1 - label], dim=1))
#         stats += confuse_matrix
#         fp_indices = torch.nonzero(torch.logical_and(1 - label, preds), as_tuple=True)[0]
#         fp_paths.extend([paths[i] for i in fp_indices])
#         fn_indices = torch.nonzero(torch.logical_and(1 - preds, label), as_tuple=True)[0]
#         fn_paths.extend([paths[i] for i in fn_indices])
#         loss_avg.update(loss.detach().cpu().item(), len(data))
#     loss = loss_avg.avg
#     print('loss:\n {}'.format(loss))
#     acc = (stats[0, 0] + stats[1, 1]) / torch.sum(stats)
#     print("acc:\n {}".format(acc))
#     print("[[tp, fp], [fn, tn]]:\n {}".format(stats))
#
#     fp_pic_num = min(len(fp_paths), 10)
#     fn_pic_num = min(len(fn_paths), 10)
#     fig = plt.figure()
#     for i in range(fp_pic_num):
#         plt.subplot(4, 5, i+1)
#         plt.imshow(imread(fp_paths[i]))
#     for i in range(fn_pic_num):
#         plt.subplot(4, 5, i+11)
#         plt.imshow(imread(fn_paths[i]))
#     plt.show()
#

if __name__ == '__main__':
    main()