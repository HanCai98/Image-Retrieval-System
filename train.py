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
        self.max_grad_norm = cfg.solver.max_grad_norm

        self.log_period = cfg.solver.log_period
        self.checkpoint_period = cfg.solver.checkpoint_period
        self.eval_period = cfg.solver.eval_period
        self.eval_loss_period = cfg.solver.eval_loss_period

        self.loss_avg = AverageMeter()  # record average loss per epoch
        self.loss_batch = 0
        self.max_acc = 0
        self.min_loss = float('inf')
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
            self.logger.info('step: {}, loss: {}'.format(self.steps, self.loss_batch))
        if self.steps % self.checkpoint_period == 0 and self.steps != 0:
            self.save()
        if self.steps % self.eval_period == 0 and self.steps != 0:
            top1_acc, top5_acc, top10_acc = self.evaluate()
            self.logger.info('Retrieve on validation:')
            self.logger.info('top1_acc, top5_acc, top10_acc: {}, {}, {}'.format(top1_acc, top5_acc, top10_acc))
            self.logger.info('-' * 20)
            self.writer.add_scalar('retrieval/top1_acc', top1_acc, self.steps)
            self.writer.add_scalar('retrieval/top5_acc', top5_acc, self.steps)
            self.writer.add_scalar('retrieval/top10_acc', top10_acc, self.steps)
        if self.steps % self.eval_loss_period == 0 and self.steps != 0:
            val_loss = self.evaluate_loss()
            self.logger.info('Validation loss: {}'.format(val_loss))
            self.writer.add_scalar('loss/validtion', val_loss, self.steps)
            if val_loss < self.min_loss:
                self.logger.info('Min loss: {}'.format(val_loss))
                self.save(True)
                self.min_loss = val_loss
            self.logger.info('-' * 20)
        self.cur_batch += 1
        self.steps += 1

    def finish_epoch(self):
        self.cur_batch = 0
        self.logger.info('Epoch {} done'.format(self.cur_epoch))
        self.logger.info('loss: {}'.format(self.loss_avg.avg))
        self.logger.info('-' * 20)
        self.cur_epoch += 1
        self.loss_avg.reset()

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        [object_positions, object_embeddings, text_ids, text_masks, img_names] = batch
        object_positions, object_embeddings = object_positions.to(self.device), object_embeddings.to(self.device)
        text_ids, text_masks = text_ids.to(self.device), text_masks.to(self.device)

        similarity_matrix = self.model(object_positions, object_embeddings, text_ids, text_masks)
        loss = self.loss_func(similarity_matrix)
        # self.logger.info('loss')
        # self.logger.info(loss.detach().cpu().numpy())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()
        self.scheduler.step()

        self.loss_batch = loss.detach().cpu().item()
        self.loss_avg.update(self.loss_batch)

    def evaluate(self):
        self.model.eval()
        image_feature_all = []
        text_feature_all = []
        text_masks_all = []
        image_names_all = []
        for batch in self.val_dl:
            [object_positions, object_embeddings, text_ids, text_masks, img_names] = batch
            object_positions, object_embeddings = object_positions.to(self.device), object_embeddings.to(self.device)
            text_ids, text_masks = text_ids.to(self.device), text_masks.to(self.device)
            image_features = self.model.image_encoder(object_positions, object_embeddings)  # [B, S, E]
            text_features = self.model.text_encoder(text_ids, text_masks)  # [B, S, E]

            image_feature_all.append(image_features.detach())
            text_feature_all.append(text_features.detach())
            text_masks_all.append(text_masks)
            image_names_all.extend(img_names)

        image_features = torch.cat(image_feature_all, dim=0)  # [N, S, E]
        text_features = torch.cat(text_feature_all, dim=0)  # [N, S, E]
        text_masks = torch.cat(text_masks_all, dim=0)  # [N, S]
        image_names = image_names_all

        similarity_scores_all = []
        for i in range(len(text_features)):
            text_feature = text_features[i]
            text_mask = text_masks[i]
            similarity_scores = self.model.interaction_model(
                image_features, torch.unsqueeze(text_feature, dim=0), torch.unsqueeze(text_mask, dim=0))  # (1, N)
            similarity_scores_all.append(similarity_scores.detach().cpu())
        scores = torch.cat(similarity_scores_all, dim=0)  # (N, N)  text -> image

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
        #     predicted_img_paths.append(image_names[int(i)])
        # results = [[a, b] for a, b in zip(image_names, predicted_img_paths)]
        # utils.save_json(results, 'retrieve_info.json')

        return top1_acc, top5_acc, top10_acc

    def evaluate_loss(self):
        self.model.eval()
        loss_am = AverageMeter()
        for batch in self.val_dl:
            [object_positions, object_embeddings, text_ids, text_masks, img_names] = batch
            object_positions, object_embeddings = object_positions.to(self.device), object_embeddings.to(self.device)
            text_ids, text_masks = text_ids.to(self.device), text_masks.to(self.device)
            similarity_matrix = self.model(object_positions, object_embeddings, text_ids, text_masks)
            loss = self.loss_func(similarity_matrix)
            loss_am.update(loss.detach().cpu().item())
        return loss_am.avg

    def save(self, is_best=False):
        if is_best:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir, 'best.pth'))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir,
                                    'checkpoint_step_{}_epoch_{}_batch_{}.pth'.format(
                                        self.steps, self.cur_epoch, self.cur_batch)))
