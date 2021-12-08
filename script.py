import json
import os

image_folder_path = "data/flickr30k-images"
caption_folder_path = "data/results_20130124.token"

image_list = os.listdir(image_folder_path)

with open(caption_folder_path, 'r') as f:
    content = f.read().split('\n')

dictionary = {}
for i in range(0, len(content), 5):
    image_name = content[i].split('#')[0]
    ls = []
    for j in range(i, i + 5):
        caption = content[j].split(maxsplit=1)[-1]
        ls.append(caption)

    ls.append(os.path.join(image_folder_path, image_name))
    dictionary[image_name] = ls
    # print(image_name)
    # print(ls)

f = open("data.json", "w")
json.dump(dictionary, f, indent=4)
f.close()

print("Successful!")