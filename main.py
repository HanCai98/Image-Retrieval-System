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
import json

import utils
from PIL import Image
from utils import setup_logger
from model import Model
from train import Trainer
from dataset import make_dataloader
from optim import make_optimizer, make_scheduler
from loss import MatrixSoftmaxCELoss, ContrastiveLoss
import matplotlib.pyplot as plt
import numpy as np
import random 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='configs/config.json',
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

    loss_func = MatrixSoftmaxCELoss(cfg.device)

    # loss_func = ContrastiveLoss(cfg.device)

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

    test_dl = make_dataloader(cfg, 'test')

    model = Model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.test.pretrained_path))

    model.eval()

    image_feature_all = []
    text_feature_all = []
    text_masks_all = []
    image_names_all = []
    for batch in test_dl:
        [object_positions, object_embeddings, text_ids, text_masks, img_names] = batch
        object_positions, object_embeddings = object_positions.to(cfg.device), object_embeddings.to(cfg.device)
        text_ids, text_masks = text_ids.to(cfg.device), text_masks.to(cfg.device)
        image_features = model.image_encoder(object_positions, object_embeddings)  # [B, S, E]
        text_features = model.text_encoder(text_ids, text_masks)  # [B, S, E]

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
        similarity_scores = model.interaction_model(
            image_features, torch.unsqueeze(text_feature, dim=0), torch.unsqueeze(text_mask, dim=0))  # (1, N)
        similarity_scores_all.append(similarity_scores.detach().cpu())
    scores = torch.cat(similarity_scores_all, dim=0)  # (N, N)  text -> image

    top10 = torch.argsort(scores, dim=1, descending=True)[:, :10]

    labels = torch.from_numpy(np.tile(np.arange(len(scores)), (10, 1))).t()
    mask = torch.eq(top10, labels).to(torch.float32)

    top1_acc = float(torch.mean(mask[:, 0]))
    top5_acc = float(5 * torch.mean(mask[:, :5]))
    top10_acc = float(10 * torch.mean(mask[:, :10]))

    # save retrieval info
    top1_indices = top10[:, 0]
    predicted_img_paths = []
    for i in top1_indices:
        predicted_img_paths.append(image_names[int(i)])
    results = [[a, b] for a, b in zip(image_names, predicted_img_paths)]
    utils.save_json(results, 'retrieve_info.json')

    print("top1_acc, top5_acc, top10_acc: {}, {}, {}".format(top1_acc, top5_acc, top10_acc))

    captions_path = "../Flickr30k-Dataset/data.json"
    images_base_path = cfg.test.images_base_path 

    for i in random.sample(range(1000), 10):
        evaluate_result(i, top10, image_names, images_base_path, captions_path, cfg.output_dir)

    # top10   # torch.tensor (1000, 10)  
    # image_names   # list of str  (1000)    like 12434218 
    # cfg.test.images_base_path    # ../Flickr30k-Dataset/flickr30k-images
    # # ../Flickr30k-Dataset/data.json

def evaluate_result(index, top10, image_names, images_base_path, captions_path, output_base_path):

    # take the index image as an example
    index_name = image_names[index] + '.jpg'
    f = open(captions_path)
    dictionary = json.load(f)
    captions = dictionary[index_name]

    candidates_list = top10[index,0:5].tolist()
    candidates_names = [name + '.jpg' for name in image_names if image_names.index(name) in candidates_list]
    i = 0
    for name in candidates_names:
        image_path = os.path.join(images_base_path, name)
        image = Image.open(image_path)
        image.save("{}/{}_{}_{}.jpg".format(output_base_path, image_names[index], name.split('.')[0], i))
        i += 1
    
    print("captions: ", captions)
    f.close()


if __name__ == '__main__':
    main()