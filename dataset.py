import logging
import os
import random

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

import utils


class PairDataset(Dataset):
    def __init__(self, dataset, num_objects, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_objects = num_objects

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_path = self.dataset[index]
        img_name = file_path.split('/')[-1].split('.')[0]
        info = utils.load_pkl(file_path)

        text = random.choice(info[:5])
        text_ids = torch.tensor(self.tokenizer.convert_to_id(text))
        text_mask = torch.where(text_ids == self.tokenizer.num_words + 1, 0, 1)

        object_positions = torch.from_numpy(np.array(info[5: 5+self.num_objects]))
        object_embeddings = torch.from_numpy(np.array(info[5+2*self.num_objects:]))

        return object_positions, object_embeddings, text_ids, text_mask, img_name


def _process_anno(path):
    file_paths = [os.path.join(path, fn) for fn in os.listdir(path)]
    return file_paths


def _make_train_loader(cfg):
    anno = _process_anno(cfg.data.train_path)
    tokenizer = utils.SimpleTokenizer(cfg.data.max_len)
    tokenizer.load_vocab(cfg.data.tokenizer_path)
    dataset = PairDataset(anno, cfg.num_objects, tokenizer)
    logger = logging.getLogger('train')
    logger.info('Total train samples: {}'.format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=cfg.solver.batch_size, num_workers=cfg.data_loader.num_workers, shuffle=True)
    return dataloader


def _make_val_loader(cfg):
    anno = _process_anno(cfg.data.val_path)
    tokenizer = utils.SimpleTokenizer(cfg.data.max_len)
    tokenizer.load_vocab(cfg.data.tokenizer_path)
    dataset = PairDataset(anno, cfg.num_objects, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def _make_test_loader(cfg):
    anno = _process_anno(cfg.data.test_path)
    tokenizer = utils.SimpleTokenizer(cfg.data.max_len)
    tokenizer.load_vocab(cfg.data.tokenizer_path)
    dataset = PairDataset(anno, cfg.num_objects, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def make_dataloader(cfg, type):
    if type == 'train':
        return _make_train_loader(cfg)
    elif type == 'validation':
        return _make_val_loader(cfg)
    else:
        return _make_test_loader(cfg)
