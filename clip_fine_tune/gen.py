import os
import utils as utils
import random


dataset_base_path = 'images'
base_output_path = 'clip_fine_tune/dataset'
if not os.path.exists(base_output_path):
    os.makedirs(base_output_path)

input_file_path = 'data.json'
dataset = utils.load_json(input_file_path)

train, validation = [], []
for img_name, text_ls in dataset.items():
    img_path = os.path.join(dataset_base_path, img_name)
    random.shuffle(text_ls)
    train.append([img_path, text_ls[:-1]])
    validation.append([img_path, [text_ls[-1]]])

utils.save_json(train, os.path.join(base_output_path, 'train.json'))
utils.save_json(validation, os.path.join(base_output_path, 'validation.json'))
