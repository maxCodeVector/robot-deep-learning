import numpy as np
import os
# import pandas as pd
import cv2
import time
from scipy.misc import imread, imresize
import pickle

file_path = "/home/hp/Documents/NUS/Mine_video_classification/UCF-101"
video_class_root = os.listdir(file_path)
x = []
y = []
image_num = 10
output = 0

for video_class_path in video_class_root[1:]:
    print(format(video_class_path))
    video_root = os.listdir(file_path + "/" + video_class_path)
    video_count = 0
    for video_path in video_root[1:]:
        video_count += 1
        video = cv2.VideoCapture(file_path + "/" + video_class_path + "/" + video_path)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        success, image = video.read()
        frame_count = 1
        while success:
            if not os.path.exists(file_path + "_modified" + "/" + video_class_path):
                os.makedirs(file_path + "_modified" + "/" + video_class_path)
            if frame_count % int(frame_length / image_num)  == 0 :
                file_modified_path = file_path + "_modified" + "/" + video_class_path + "/" + str(video_count) + "_" +  str(frame_count / int(frame_length / image_num) ) + ".jpg"
                image = cv2.resize(image,(112,112))
                cv2.imwrite( file_modified_path , image)
                x.append(image)
                y.append(output)
            success, image = video.read()
            frame_count += 1
            #print(str(frame_count) + "_" + str(video_count))
    output += 1
    print("One video finished")
x = np.array(x)
y = np.array(y)
print("x", len(x), len(y))
