import json
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

## aihub 데이터 annotation을 읽어서 단어 단위로 잘라서 data에 저장하기

data_root_path = 'D:/korean_hand_write/01.handwrite/1_word/'
save_root_path = 'D:/study_data/_data/deep-text-recognition-benchmark/data/'

test_annotations = json.load(open('C:/team_project/test_annotation.json'))
gt_file = open(save_root_path+'gt_test.txt', 'w')
for file_name in tqdm(test_annotations):
    annotations = test_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)
    text = annotations[0]['text']
    cv2.imwrite(save_root_path+'test/'+file_name, image)
    gt_file.write("test/{}\t{}\n".format(file_name, text))

validation_annotations = json.load(open('C:/team_project/validation_annotation.json'))
gt_file = open(save_root_path+'gt_validation.txt', 'w')
for file_name in tqdm(validation_annotations):
    annotations = validation_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)
    text = annotations[0]['text']
    cv2.imwrite(save_root_path+'validation/'+file_name, image)
    gt_file.write("validation/{}\t{}\n".format(file_name, text))    
        
train_annotations = json.load(open('C:/team_project/train_annotation.json'))
gt_file = open(save_root_path+'gt_train.txt', 'w')
for file_name in tqdm(train_annotations):
    annotations = train_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)
    text = annotations[0]['text']
    cv2.imwrite(save_root_path+'train/'+file_name, image)
    gt_file.write("train/{}\t{}\n".format(file_name, text))