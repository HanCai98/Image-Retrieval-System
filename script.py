import os
import json
import random
from inference import gen_one_image, load
from tqdm import tqdm
import torch

def save_dic(dic, path):
    with open(path, 'w') as f:
        json.dump(dic, f, indent=4)


def load_dic(path):
    with open(path, 'rb') as f:
        dic = json.load(f)
    return dic


def get_images(path, num=1000):
    clip_model, preprocess, tokenizer, model = load('transformer.pt')
    fn_list = list(os.listdir(path))
    random.shuffle(fn_list)
    fn_list = fn_list[:num]
    res = {}
    for fn in tqdm(fn_list):
        res[fn] = gen_one_image(clip_model, preprocess, tokenizer, model, os.path.join(path, fn))
    save_dic(res, 'data.json')


if __name__ == '__main__':
    get_images('images')
    # a = 10
    # x = torch.tensor([1, 2, 3])
    # x = torch.tile(x, (a, 1)).reshape((-1, a))
    # print(x.shape)
