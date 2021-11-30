import logging
import random

import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from torchvision.io import read_image
import utils as utils


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, texts = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, random.choice(texts), img_path


def _process_anno(path):
    data = utils.load_json(path)  # list of [img_path, text]
    return data


def _make_train_loader(cfg, preprocess):
    trm = T.Compose([
        T.ToPILImage(),
        preprocess,
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        AddGaussianNoise(0.1, 0.08),
    ])
    anno = _process_anno(cfg.data.train_path)
    dataset = ImageDataset(anno, trm)
    logger = logging.getLogger('train')
    logger.info('Total train samples: {}'.format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=cfg.solver.batch_size, num_workers=cfg.data_loader.num_workers)
    return dataloader


def _make_val_loader(cfg, preprocess):
    trm = T.Compose([
        T.ToPILImage(),
        preprocess,
    ])
    anno = _process_anno(cfg.data.val_path)
    dataset = ImageDataset(anno, trm)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def _make_test_loader(cfg, preprocess):
    trm = T.Compose([
        T.ToPILImage(),
        preprocess,
    ])
    anno = _process_anno(cfg.data.test_path)
    dataset = ImageDataset(anno, trm)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.test.num_workers)
    return dataloader


def make_dataloader(cfg, type, preprocess):
    if type == 'train':
        return _make_train_loader(cfg, preprocess)
    elif type == 'validation':
        return _make_val_loader(cfg, preprocess)
    else:
        return _make_test_loader(cfg, preprocess)


