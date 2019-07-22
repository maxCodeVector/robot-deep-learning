import os
import shutil
import numpy as np


if __name__ == '__main__':
    rootPath = "/home/hya/Downloads/20bn-jester-v1"
    label_dict = {}
    with open("jester-v1-labels.csv", "r") as flabel:
        label_list = flabel.readlines()
        for i, label in enumerate(label_list):
            label_dict[label] = i
    ground_truth_list = np.zeros(150000, dtype=np.int)
    with open("jester-v1-train.csv", "r") as fdata:
        info = fdata.readlines()
        for i, datalabel in enumerate(info):
            data = datalabel.split(";")
            ground_truth_list[int(data[0])] = label_dict[data[1]]
    type_num = [0] * 28
    for file in os.listdir(rootPath):
        type_num[ground_truth_list[int(file)]+1] += 1

    print(type_num)
    print("complete")