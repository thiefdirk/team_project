import random
import os
import json
import sys
import time
import datetime
import tqdm as tqdm


import json
file = json.load(open('D:\korean_hand_write/01.handwrite/handwriting_data_info_clean.json', "rt", encoding='UTF8'))

annotation = [a for a in file['annotations'] if a['attributes']['type']=='단어(어절)']

print(len(annotation))
print(annotation[0])

# 'type' == '단어(어절)' 인것만 추출
# file = [x for x in file if x['attributes']['type'] == '단어(어절)']



ocr_good_files = os.listdir('D:\korean_hand_write/01.handwrite/1_word/')
len(ocr_good_files) # 37220

random.shuffle(ocr_good_files)

n_train = int(len(ocr_good_files) * 0.7)
n_validation = int(len(ocr_good_files) * 0.15)
n_test = int(len(ocr_good_files) * 0.15)

print(n_train, n_validation, n_test) # 26054 5583 5583

train_files = ocr_good_files[:n_train]
validation_files = ocr_good_files[n_train: n_train+n_validation]
test_files = ocr_good_files[-n_test:]

## train/validation/test 이미지들에 해당하는 id 값을 저장

train_img_ids = {}
validation_img_ids = {}
test_img_ids = {}

for image in tqdm.tqdm(file['images']):
    if image['file_name'] in train_files:
        train_img_ids[image['file_name']] = image['id']
    elif image['file_name'] in validation_files:
        validation_img_ids[image['file_name']] = image['id']
    elif image['file_name'] in test_files:
        test_img_ids[image['file_name']] = image['id']

## train/validation/test 이미지들에 해당하는 annotation 들을 저장

train_annotations = {f:[] for f in train_img_ids.keys()}
validation_annotations = {f:[] for f in validation_img_ids.keys()}
test_annotations = {f:[] for f in test_img_ids.keys()}

train_ids_img = {train_img_ids[id_]:id_ for id_ in train_img_ids}
validation_ids_img = {validation_img_ids[id_]:id_ for id_ in validation_img_ids}
test_ids_img = {test_img_ids[id_]:id_ for id_ in test_img_ids}

for idx, annotation in tqdm.tqdm(enumerate(file['annotations'])):
    if idx % 5000 == 0:
        print(idx,'/',len(file['annotations']),'processed')
    if annotation['attributes']['type'] != '단어(어절)':
        continue
    if annotation['image_id'] in train_ids_img:
        train_annotations[train_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in validation_ids_img:
        validation_annotations[validation_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in test_ids_img:
        test_annotations[test_ids_img[annotation['image_id']]].append(annotation)

with open('train_annotation.json', 'w') as file:
    json.dump(train_annotations, file)
with open('validation_annotation.json', 'w') as file:
    json.dump(validation_annotations, file)
with open('test_annotation.json', 'w') as file:
    json.dump(test_annotations, file)