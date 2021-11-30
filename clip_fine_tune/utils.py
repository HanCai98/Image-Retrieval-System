import json
import numpy as np
import logging
import os
import dotmap

def load_json(filepath):
    with open(filepath, 'r') as f:
        dic = json.load(f)
    return dic


def save_json(dic, filepath):
    with open(filepath, 'w') as f:
        json.dump(dic, f, indent=4)


def process_cfg(cfg_path):
    cfg_dict = load_json(cfg_path)
    cfg = dotmap.DotMap(cfg_dict)
    return cfg


def setup_logger(save_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(save_dir, 'exp.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count