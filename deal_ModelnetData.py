import os
import sys
import h5py
import numpy as np


# 将数据写入h5py文件
def write_data2h5file(file_name, h5data, label=None):
    h5f = h5py.File(file_name, 'w')
    h5f['label'] = label
    h5f['data'] = h5data
    h5f.close()


# Write numpy array data and label to h5_filename
def save_h5_data_label(h5_filename, data, label=None, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('data', data=data, compression='gzip', compression_opts=4, dtype=data_dtype)
    h5_fout.create_dataset('label', data=label, compression='gzip', compression_opts=1, dtype=label_dtype)
    h5_fout.close()


# return the files' name in current dir
def getFiles_name(file_dir):
    for root, dirs, files_name in os.walk(file_dir):
        num = 1
    return files_name


# Sampling the points by farthest_point method
def farthest_point_sample(xyz, npoint):
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)  # random select one as centroid
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)  # select the farthest one as centroid
        # print('index:%d, dis:%.3f'%(farthest,np.max(distance)))
    return centroids


# Read the data from *.off file
def read_off(filename):
    f = open(filename)
    f.readline()         # ignore the 'OFF' at the first line
    f.readline()         # ignore the second line
    coord2points = []
    while True:
        new_line = f.readline()
        x = new_line.split(' ')
        # print('x:', x)
        if x[0] != '3':   # x[0]==3 代表三角面的三个顶点
            coord2point = np.array(x[0:3], dtype='float32')
            coord2points.append(coord2point)
        else:
            break
    f.close()

    num2points = 2048  # we need sample 2048 points from current point cloud
    # if the numbers of points are less than 2000, extent the point set
    if len(coord2points) < (num2points + 3):
        # print("none")
        return None
    coord2points = np.array(coord2points)
    # take and shuffle points
    # index = np.random.choice(len(All_points), num_select, replace=False)
    xyz = coord2points.copy()
    points_farthest_index = farthest_point_sample(xyz, num2points).astype(np.int64)
    points_farthest = xyz[points_farthest_index, :]
    centroid = np.mean(points_farthest, axis=0)
    points_unit_sphere = points_farthest - centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points_farthest) ** 2, axis=-1)))
    points_unit_sphere /= furthest_distance
    # print('*************************')
    return list(points_unit_sphere)           # return N*3 array 的list


def make_trian_Dataset(name2originData=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print('Base_dir:', BASE_DIR)
    print('\n')
    sys.path.clear()
    sys.path.append(BASE_DIR)
    print('Contents for path:', sys.path)

    filename_train0 = BASE_DIR + '/data/mydata/ply_data_train0.h5'  # 创建点云的路径
    filename_train1 = BASE_DIR + '/data/mydata/ply_data_train1.h5'  # 创建点云的路径
    filename_train2 = BASE_DIR + '/data/mydata/ply_data_train2.h5'  # 创建点云的路径
    filename_train3 = BASE_DIR + '/data/mydata/ply_data_train3.h5'  # 创建点云的路径
    filename_train4 = BASE_DIR + '/data/mydata/ply_data_train4.h5'  # 创建点云的路径

    BasePath2data = '%s\%s' % ('data', str(name2originData))
    CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
               'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    label_list2class = []
    traindata_list2class = []
    for current_label, class_name in enumerate(CLASSES):
        print(current_label)
        print(class_name)
        data_dir = os.path.join(BasePath2data, class_name, 'train')
        print(data_dir)
        if os.path.exists(data_dir) != True:
            break
        filesName_list = getFiles_name(data_dir)  # 得到文件夹下的所有文件的名称
        for fileName in filesName_list:
            if str.lower(fileName) == '.ds_store':
                continue
            data_filename = os.path.join(data_dir, fileName)
            # data_filename = os.path.join(data_dir, fileName)
            print(data_filename)
            current_data = read_off(data_filename)
            if current_data == None:
                continue
            label_list2class.append(current_label)
            traindata_list2class.append(current_data)

        if current_label == 1:
            save_h5_data_label(filename_train0, traindata_list2class, label_list2class)
            traindata_list2class.clear()
            label_list2class.clear()
        elif current_label == 3:
            save_h5_data_label(filename_train1, traindata_list2class, label_list2class)
            traindata_list2class.clear()
            label_list2class.clear()
        elif current_label == 5:
            save_h5_data_label(filename_train2, traindata_list2class, label_list2class)
            traindata_list2class.clear()
            label_list2class.clear()
        elif current_label == 7:
            save_h5_data_label(filename_train3, traindata_list2class, label_list2class)
            traindata_list2class.clear()
            label_list2class.clear()
        elif current_label == 9:
            save_h5_data_label(filename_train4, traindata_list2class, label_list2class)
            traindata_list2class.clear()
            label_list2class.clear()


def make_test_Dataset(name2originData=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print('Base_dir:', BASE_DIR)
    print('\n')
    sys.path.clear()
    sys.path.append(BASE_DIR)
    print('Contents for path:', sys.path)

    filename_test0 = BASE_DIR + '/data/mydata/ply_data_test0.h5'  # 创建点云的路径
    filename_test1 = BASE_DIR + '/data/mydata/ply_data_test1.h5'  # 创建点云的路径

    BasePath2data = '%s\%s' % ('data', str(name2originData))
    print('BasePath2data:', BasePath2data)
    CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
               'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    label_list2class = []
    testdata_list2class = []
    for current_label, class_name in enumerate(CLASSES):
        print(current_label)
        print(class_name)
        data_dir = os.path.join(BasePath2data, class_name, 'test')
        print(data_dir)
        if os.path.exists(data_dir) != True:
            break
        filesName_list = getFiles_name(data_dir)  # 得到文件夹下的所有文件的名称
        for fileName in filesName_list:
            if str.lower(fileName) == '.ds_store':
                continue
            data_filename = os.path.join(data_dir, fileName)
            print(data_filename)
            current_data = read_off(data_filename)
            if current_data == None:
                continue
            label_list2class.append(current_label)
            testdata_list2class.append(current_data)

        if current_label == 4:
            save_h5_data_label(filename_test0, testdata_list2class, label_list2class)
            testdata_list2class.clear()
            label_list2class.clear()
        elif current_label == 9:
            save_h5_data_label(filename_test1, testdata_list2class, label_list2class)
            testdata_list2class.clear()
            label_list2class.clear()


if __name__ == "__main__":
    # # filePath = 'data/ModelNet10/bathtub/train'
    # # getFiles_name(filePath)
    # filename = 'data/ModelNet10/bathtub/train/bathtub_0001.off'
    # # filename = 'bathtub_0001.off'
    # data = read_off(filename)

    dataName = 'ModelNet10'
    make_trian_Dataset(name2originData=dataName)
    make_test_Dataset(name2originData=dataName)

