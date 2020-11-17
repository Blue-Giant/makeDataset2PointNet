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


# Write numpy array data and label and numpy array normal to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal,
                              data_dtype='float32', label_dtype='uint8', normal_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5_data_label(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# return the files' name in current dir
def getFiles_name(file_dir):
    for root, dirs, files_name in os.walk(file_dir):
        num = 1
    return files_name


# Sampling the points by random method, the return the sampling points
def random_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        samplePoint: sampled points from the current cloud, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    flagArray = np.concatenate((np.ones(npoint), np.zeros(N-npoint)), axis=-1)
    np.random.shuffle(flagArray)
    indexes2Points = np.arange(1, N+1, 1, np.int32)
    flag2Points = np.multiply(flagArray.astype(np.int32), indexes2Points)
    indexes2samplePoints = []
    for index in range(N):
        if flag2Points[index] != 0:
            indexes2samplePoints.append(flag2Points[index])
    indexes2samples = np.array(indexes2samplePoints) - 1
    samplePoint = xyz[indexes2samples.astype(np.int32)]
    return samplePoint


# Sampling the points by uniform method, the return the sampling points
def uniform_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        samplePoint: sampled points from the current cloud, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    indexes2samplePoints = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        indexes2samplePoints[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    samplePoint = point[indexes2samplePoints.astype(np.int32)]
    return samplePoint


# Sampling the points by farthest_point method, the return the sampling points
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        samplePoint: sampled points from the current cloud, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    indexes2samplePoints = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        indexes2samplePoints[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    samplePoint = point[indexes2samplePoints.astype(np.int32)]
    return samplePoint


# Sampling the points by farthest_point method, the return the indexes of sampling points
def getIndexes_farthest_point_sample(xyz, npoint):
    """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
             indexes2samplePoints
        """
    N, C = xyz.shape
    indexes2samplePoints = np.zeros(npoint)
    distance = np.ones(N) * 1e10                         # initiate the farthest distance array
    index2farthest = np.random.randint(0, N)             # random select one as initial centroid
    for i in range(npoint):
        indexes2samplePoints[i] = index2farthest         # add the index of farthest points into indexes
        centroid = xyz[index2farthest, :].reshape(1, 3)  # obtain the coords for centroid
        dist = np.sum((xyz - centroid) ** 2, -1)         # calculating the distance between centroid and other points
        mask = dist < distance
        distance[mask] = dist[mask]                      # update the farthest distance array by current results
        index2farthest = np.argmax(distance)             # select the farthest one as new centroid
    return indexes2samplePoints


# Get the data from *.off file
def getData_from_Off(filename, opt2sample=None, normalize_PointSet=False):
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
    # if the numbers of points are less than 2048, extent the point set
    if len(coord2points) < num2points:
        # print("none")
        return None
    coord2points = np.array(coord2points)
    xyz = coord2points.copy()
    if 'farthest_point_sample'== str.lower(opt2sample):
        points_sample = farthest_point_sample(xyz, num2points).astype(np.int64)
    elif 'random_point_sample'== str.lower(opt2sample):
        points_sample = random_point_sample(xyz, num2points)
    else:
        points_sample = uniform_point_sample(xyz, num2points)
    print('points_sample:', points_sample)

    centroid = np.mean(points_sample, axis=0)
    points_unit_sphere = points_sample - centroid
    if (normalize_PointSet):
        # centroid = np.mean(points_sample, axis=0)
        # points_unit_sphere = points_sample - centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points_unit_sphere) ** 2, axis=-1)))
        points_unit_sphere /= furthest_distance
    return points_unit_sphere


def make_trian_Dataset(name2originData=None, sample_opt=None):
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
            current_data = getData_from_Off(data_filename, opt2sample=sample_opt, normalize_PointSet=True)
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


def make_test_Dataset(name2originData=None, sample_opt=None):
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
            current_data = getData_from_Off(data_filename, opt2sample=sample_opt, normalize_PointSet=True)
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
    # data2points = getData_from_Off(filename, normalize_PointSet=False)
    # print('data2points:', data2points)

    dataName = 'ModelNet10'
    option2sample = 'random_point_sample'
    make_trian_Dataset(name2originData=dataName, sample_opt=option2sample)
    make_test_Dataset(name2originData=dataName, sample_opt=option2sample)

