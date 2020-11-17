import os
import sys
import numpy as np
import h5py
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.clear()
sys.path.append(BASE_DIR)
print(sys.path)
print(BASE_DIR)


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    h5f = h5py.File(h5_filename)
    data = h5f['data'][:]
    label = h5f['label'][:]
    normal = h5f['normal'][:]
    return data, label, normal


def loadDataFile(filename):
    return load_h5(filename)


filename_train0 = BASE_DIR + '/data/mydata/train0.h5'  # 创建点云的路径
filename_train1 = BASE_DIR + '/data/mydata/train1.h5'  # 创建点云的路径
filename_train2 = BASE_DIR + '/data/mydata/train2.h5'  # 创建点云的路径

filename_test0 = BASE_DIR + '/data/mydata/test0.h5'  # 创建点云的路径
filename_test1 = BASE_DIR + '/data/mydata/test1.h5'  # 创建点云的路径


current_data, current_label, current_normal = loadDataFile(filename_train0)
current_label = np.squeeze(current_label)
label_list2sub_class = []
data_list2sub_class = []
normal_list2sub_class = []
label_length = len(current_label)
for j in range(label_length):
    data = current_data[j]
    normal = current_normal[j]
    label_list2sub_class.append(current_label[j])
    data_list2sub_class.append(current_data[j])
    normal_list2sub_class.append(current_normal[j])

print('ok')


