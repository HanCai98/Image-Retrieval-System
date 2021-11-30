import clip
import torch
import os

import torch
from tqdm import tqdm
import numpy as np
import logging
import os
from optim import make_optimizer
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter


class Trainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, optimizer, scheduler, loss_func):
        self.cfg = cfg

        self.device = cfg.device
        self.model = model.to(cfg.device)
        self.train_dl = train_dl
        self.val_dl = val_dl

        self.epochs = cfg.solver.num_epochs
        self.optim = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

        self.log_period = cfg.solver.log_period
        self.checkpoint_period = cfg.solver.checkpoint_period
        self.eval_period = cfg.solver.eval_period

        self.loss_avg = AverageMeter()
        self.loss_batch = 0
        self.max_acc = 0
        self.cur_epoch = 0  # start from 0
        self.cur_batch = 0  # start from 0
        self.steps = 0  # total steps

        self.output_dir = cfg.output_dir
        self.writer = SummaryWriter(self.output_dir)
        self.logger = logging.getLogger('train')

        self.logger.info('Trainer Built.')

    def train(self):
        for epoch in range(self.epochs):
            for batch in tqdm(self.train_dl):
                self.step(batch)
                self.finish_batch()
            self.finish_epoch()

    def finish_batch(self):
        if self.steps % self.log_period == 0 and self.steps != 0:
            self.writer.add_scalar('loss/train', self.loss_batch, self.steps)
        if self.steps % self.checkpoint_period == 0 and self.steps != 0:
            self.save()
        if self.steps % self.eval_period == 0 and self.steps != 0:
            top1_acc, top5_acc, top10_acc = self.evaluate()
            self.logger.info('Validation Result:')
            self.logger.info('top1_acc, top5_acc, top10_acc: {}, {}, {}'.format(top1_acc, top5_acc, top10_acc))
            self.logger.info('-' * 20)
            self.writer.add_scalar('loss/top1_acc', top1_acc, self.steps)
            self.writer.add_scalar('loss/top5_acc', top5_acc, self.steps)
            self.writer.add_scalar('loss/top10_acc', top10_acc, self.steps)
            if top5_acc > self.max_acc:
                self.logger.info('Best top5_acc: {}', top5_acc)
                self.save(True)
        self.cur_batch += 1
        self.steps += 1

    def finish_epoch(self):
        self.cur_batch = 0
        self.logger.info('Epoch {} done'.format(self.cur_epoch))
        self.logger.info('loss: {}'.format(self.loss_avg.avg))
        self.logger.info('-' * 20)
        self.cur_epoch += 1

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        [images, texts, img_path] = batch
        texts = clip.tokenize(texts)
        images, texts = images.to(self.device), texts.to(self.device)
        logits_images, logits_texts = self.model(images, texts)
        loss = self.loss_func(logits_images, logits_texts)
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        self.loss_batch = loss.cpu().item()
        self.loss_avg.update(self.loss_batch)

    def evaluate(self):
        self.model.eval()
        image_feature_ls = []
        text_feature_ls = []
        image_path_ls = []
        for images, texts, img_paths in self.val_dl:
            texts = clip.tokenize(texts)
            images, texts = images.to(self.device), texts.to(self.device)
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_feature_ls.append(image_features)
            text_feature_ls.append(text_features)
            image_path_ls.extend(img_paths)

        image_features = torch.concat(image_feature_ls, dim=0)
        text_features = torch.concat(text_feature_ls, dim=0)

        score_ls = []
        for image_feature in image_features:
            scores = torch.matmul(torch.unsqueeze(image_feature, 0), text_features)
            score_ls.append(scores)
        scores = torch.concat(score_ls, dim=0)
        top10 = torch.argsort(scores, dim=1)[:, :10]

        labels = torch.tile(torch.arange(len(scores)), (10, 1)).reshape(-1, 10)
        mask = torch.eq(top10, labels).to(torch.int)

        top1_acc = float(torch.mean(mask[:, 0]))
        top5_acc = float(torch.mean(mask[:, :5]))
        top10_acc = float(torch.mean(mask[:, :10]))

        return top1_acc, top5_acc, top10_acc

    def save(self, is_best=False):
        if is_best:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir, 'best.pth'))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir,
                                    'checkpoint_step_{}_epoch_{}_batch_{}.pth'.format(
                                        self.steps, self.cur_epoch, self.cur_batch)))
