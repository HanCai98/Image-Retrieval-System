import json
import numpy as np
import logging
import os
import dotmap
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle


class SimpleTokenizer:
    def __init__(self, max_len=30):
        self.lemmatizer = WordNetLemmatizer()
        self.word2id = {}  # only contain known words
        self.num_words = 0  # does not include <PAD> and <UNKNOWN>.
        # <UNKNOWN> index: self.num_word; <PAD> index: self.num_word + 1
        self.max_len = max_len

    def build_vocab(self, texts, min_freq=5, save_path='vocab.json'):
        '''
        :param texts: list of list of word.
        :param min_freq: if the frequency of word is less than min_freq, it will be identified as <UNKNOWN>
        '''
        word_freq = {}
        for text in texts:
            tokens = [self.lemmatizer.lemmatize(w.lower()) for w in text]
            for word in tokens:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        word2id = {}
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in word2id:
                word2id[word] = len(word2id)
        self.num_words = len(word2id)
        self.word2id = word2id
        save_json(self.word2id, save_path)

    def load_vocab(self, path):
        self.word2id = load_json(path)
        self.num_words = len(self.word2id)

    def convert_to_id(self, sentence):
        '''
        :param sentence: str (split by whitespace)
        :return: list of int
        '''
        tokens = [self.lemmatizer.lemmatize(word) for word in sentence.lower().split(' ')]
        ids = [self.num_words + 1] * self.max_len  # initialize with <PAD>
        for i, word in enumerate(tokens):
            if i >= self.max_len:
                break
            if word in self.word2id:
                ids[i] = self.word2id[word]
            else:
                ids[i] = self.num_words
        return ids


def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


if __name__ == '__main__':
    texts = []
    dic = load_json('../Flickr30k-Dataset/train.json')
    for img_name, info in dic.items():
        for text in info[:-1]:
            texts.append(text.split(' '))
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, 5, 'vocab.json')
    count = 0
    for img_name, info in dic.items():
        count += 1
        if count >= 10:
            break
        for text in info[:-1]:
            print(tokenizer.convert_to_id(text))
