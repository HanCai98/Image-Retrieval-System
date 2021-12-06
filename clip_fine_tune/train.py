import clip
import torch
import os

import torch
from tqdm import tqdm
import numpy as np
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter
import utils


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
        if self.steps % self.eval_period == 0:
            top1_acc, top5_acc, top10_acc = self.evaluate()
            self.logger.info('Validation Result:')
            self.logger.info('top1_acc, top5_acc, top10_acc: {}, {}, {}'.format(top1_acc, top5_acc, top10_acc))
            self.logger.info('-' * 20)
            self.writer.add_scalar('loss/top1_acc', top1_acc, self.steps)
            self.writer.add_scalar('loss/top5_acc', top5_acc, self.steps)
            self.writer.add_scalar('loss/top10_acc', top10_acc, self.steps)
            if top5_acc > self.max_acc:
                self.logger.info('Best top5_acc: {}'.format(top5_acc))
                self.save(True)
                self.max_acc = top5_acc
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
        self.model.logit_scale.requires_grad = False
        self.model.transformer.train(False)
        self.optim.zero_grad()
        [images, texts, img_path] = batch
        # self.logger.info('img_path')
        # self.logger.info(img_path)
        # self.logger.info('texts')
        # self.logger.info(texts)
        texts = clip.tokenize(texts)
        images, texts = images.to(self.device), texts.to(self.device)
        logits_per_image, logits_per_texts = self.model(images, texts)
        probs_per_image, probs_per_text = torch.softmax(logits_per_image, dim=-1), torch.softmax(logits_per_texts, dim=-1)
        # self.logger.info('probs_images')
        # self.logger.info(probs_per_image.detach().cpu().numpy())
        # self.logger.info('probs_texts')
        # self.logger.info(probs_per_text.detach().cpu().numpy())
        loss = self.loss_func(probs_per_image, probs_per_text)
        self.logger.info('loss')
        self.logger.info(loss.detach().cpu().numpy())
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        del images, texts, logits_per_image, logits_per_texts, probs_per_image, probs_per_text
        # del images, texts
        self.loss_batch = loss.detach().cpu().item()
        self.loss_avg.update(self.loss_batch)

    def evaluate(self):
        # torch.cuda.empty_cache()
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

            image_feature_ls.append(image_features.detach().cpu())
            text_feature_ls.append(text_features.detach().cpu())
            image_path_ls.extend(img_paths)

            del images, texts, image_features, text_features

        image_features = torch.cat(image_feature_ls, dim=0)
        text_features = torch.cat(text_feature_ls, dim=0)
        img_paths = image_path_ls

        scores = torch.matmul(text_features, image_features.t())
        top10 = torch.argsort(scores, dim=1, descending=True)[:, :10]

        labels = torch.from_numpy(np.tile(np.arange(len(scores)), (10, 1))).t()
        mask = torch.eq(top10, labels).to(torch.float32)

        top1_acc = float(torch.mean(mask[:, 0]))
        top5_acc = float(5 * torch.mean(mask[:, :5]))
        top10_acc = float(10 * torch.mean(mask[:, :10]))

        # # save retrieval info
        # top1_indices = top10[:, 0]
        # predicted_img_paths = []
        # for i in top1_indices:
        #     predicted_img_paths.append(img_paths[int(i)])
        # results = [[a, b] for a, b in zip(img_paths, predicted_img_paths)]
        # utils.save_json(results, 'retrieve_info.json')

        # torch.cuda.empty_cache()
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
