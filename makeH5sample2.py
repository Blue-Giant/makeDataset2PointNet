import os
import sys
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print('Base_dir:', BASE_DIR)
print('\n')
sys.path.clear()
sys.path.append(BASE_DIR)
print('Contents for path:', sys.path)

# label2sub_class = (1, 2, 3, 4, 5, 6, 7, 8, 9)
label2sub_class = (2, 4, 8, 12, 14, 19, 28, 30, 33, 38)


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5data(h5_filename):
    h5f = h5py.File(h5_filename)
    data = h5f['data'][:]
    label = h5f['label'][:]
    normal = h5f['normal'][:]
    return data, label, normal


def loadDataFile(filename):
    return load_h5data(filename)


# 写入文件, 如果要存放的数据为
def write_data2h5file(file_name, label, h5data, h5normal):
    h5f = h5py.File(file_name, 'w')
    h5f['label'] = label
    h5f['data'] = h5data
    h5f['normal'] = h5normal
    h5f.close()


# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal, data_dtype='float32', label_dtype='uint8',
                              normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('normal', data=normal, compression='gzip', compression_opts=4, dtype=normal_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()


# 由于 train_files 的路径设定为 .../data/modelnet40_ply_hdf5_2048/...',在这里我们不做修改
TRAIN_FILES = getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# 本人的存储路径，modelnet40_ply_hdf5_2048有5个train文件，在这里也是5个。两个test文件，本人如是
# ...//data/mydata/...为本人存放数据的路径
filename_train0 = BASE_DIR + '/data/mydata/ply_data_train0.h5'  # 创建点云的路径
filename_train1 = BASE_DIR + '/data/mydata/ply_data_train1.h5'  # 创建点云的路径
filename_train2 = BASE_DIR + '/data/mydata/ply_data_train2.h5'  # 创建点云的路径
filename_train3 = BASE_DIR + '/data/mydata/ply_data_train3.h5'  # 创建点云的路径
filename_train4 = BASE_DIR + '/data/mydata/ply_data_train4.h5'  # 创建点云的路径

filename_test0 = BASE_DIR + '/data/mydata/ply_data_test0.h5'  # 创建点云的路径
filename_test1 = BASE_DIR + '/data/mydata/ply_data_test1.h5'  # 创建点云的路径

file_length2train = len(TRAIN_FILES)
file_length2test = len(TEST_FILES)

label_list2sub_class = []
data_list2sub_class = []
normal_list2sub_class = []
for fn in range(len(TRAIN_FILES)):
    current_data, current_label, current_normal = loadDataFile(TRAIN_FILES[fn])
    current_label = np.squeeze(current_label)
    label_length = len(current_label)
    for j in range(label_length):
        label = current_label[j]
        if label in label2sub_class:
            data = current_data[j]
            normal = current_normal[j]
            label_list2sub_class.append(current_label[j])
            data_list2sub_class.append(current_data[j])
            normal_list2sub_class.append(current_normal[j])

    if fn == 0:
        save_h5_data_label_normal(filename_train0, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    elif fn == 1:
        save_h5_data_label_normal(filename_train1, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    elif fn == 2:
        save_h5_data_label_normal(filename_train2, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    elif fn == 3:
        save_h5_data_label_normal(filename_train3, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    elif fn == 4:
        save_h5_data_label_normal(filename_train4, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    label_list2sub_class.clear()
    data_list2sub_class.clear()
    normal_list2sub_class.clear()


for fn in range(len(TEST_FILES)):
    current_data, current_label, current_normal = loadDataFile(TEST_FILES[fn])
    current_label = np.squeeze(current_label)
    label_length2test = len(current_label)
    for j in range(label_length2test):
        label = current_label[j]
        if label in label2sub_class:
            label_list2sub_class.append(current_label[j])
            data_list2sub_class.append(current_data[j])
            normal_list2sub_class.append(current_normal[j])

    if fn == 0:
        save_h5_data_label_normal(filename_test0, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    else:
        save_h5_data_label_normal(filename_test1, data_list2sub_class, label_list2sub_class, normal_list2sub_class)
    label_list2sub_class.clear()
    data_list2sub_class.clear()
    normal_list2sub_class.clear()


