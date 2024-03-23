import os

import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

EPOCHS = 50  # 300左右
INIT_LR = 1e-5  # 学习率，一般情况从0.001开始逐渐降低，也别太小了到1e-6就可以了。
batch_size = 32  # 根据硬件的情况和数据集的大小设置，太小了抖的厉害，太大了收敛不好.
classnum = 4  # 类别数
# c = 1  # 挂网
c = 2  # 地埋

# ----------------------------------------------融合---------------------------------------
# d = 3


# ----------------------------------------------一维时域信号---------------------------------------
d = 1  # 一维时域信号
# ------------------挂网光纤--------------
# datapath1 = 'E:\\VD\\Compare\\datas\\0811-GW-Datas.csv'
# datapath1 = 'F:/1204/Compare/datas/.csv'
# datapath1 = 'F:/1204/Compare/datas/0904-datas.csv'
# datapath1 = 'F:/1204/Compare/datas/0811-GW-Datas.csv'
# -----------------地埋光缆----------------
# datapath1 = 'F:/1204/Compare/datas/DM-0814-Datas.csv'
datapath1 = 'F:/1204/Compare/datas/DM-0816.csv'
# datapath1 = 'F:/1204/Compare/datas/DM-Features-train-1D.csv'
# datapath1 = 'F:/1204/Compare/datas/DM-0814-Features.csv'

# 加载训练集数据
data1 = pd.read_csv(datapath1, header=None)
# 训练集数据归一化
# data1 = data1 / 32768
df_norm = data1[data1.columns[0:2000]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# df_norm = data1[data1.columns[0:26]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
input_1d_data = np.array(df_norm)
# input_1d_data = input_1d_data.reshape(4000, 2000, 1)
input_1d_data = input_1d_data.reshape(4000, 26, 1)
print('input_1d_data:--------', input_1d_data.shape)  # 4000*2000

# ---------------------------------------------时频图-----------------------------------------
# d = 2
# ------------------挂网光纤--------------
# datapath2 = 'E:/VD/GW/CWT'  # 噪声、持械、抛物、晃动
# datapath2 = 'F:/1204/Compare/datas/GW-CWT'   # 噪声、攀爬、抛物、晃动
# datapath2 = 'F:/1204/Compare/datas/GW-CWT-Gray' # 噪声、攀爬、抛物、晃动 灰度图
# datapath2 = 'F:/1204/Compare/datas/GW-CWT-0904'
# datapath2 = 'F:/1204/Compare/datas/GW-CWT-0811'
# -----------------地埋光缆----------------
datapath2 = 'F:/1204/Compare/datas/DM-CWT-0814'


dicClass = {'0': 0, '1': 1, '2': 2, '3': 3}
labelList_train = []  # 类别数量，数据集有4个类别，所有就分为4类。


# out_path = 'F:/1204/Compare/datas/GW-CWT-Gray'
# convert_color(datapath2, out_path)

def loadImageData_train():
    imageList = []
    listImage = os.listdir(datapath2)
    for img in listImage:
        # listImage.sort(key=lambda x: int(x.split('-')[0]))  # 排序
        labelName = dicClass[img.split('-')[0]]
        # print('labelName:+++++++++++/n', labelName)
        labelList_train.append(labelName)
        dataImgPath = os.path.join(datapath2, img)
        # print(dataImgPath)
        image = cv2.imdecode(np.fromfile(dataImgPath, dtype=np.uint8), -1)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        image = img_to_array(image)
        imageList.append(image)
    imageList = np.array(imageList, dtype="int") / 255.0
    return imageList


print("开始加载训练集数据")
input_img_data = loadImageData_train()
image_labels = np.array(labelList_train)
# image_labels = image_labels.reshape((4000, 1))
# print('input_img_data:----------', input_img_data.shape)  # 4000*224*224*3
# print('input_img_data:++++++++++', input_img_data)

# 划分数据集
# signal_train, signal_val, image_train, image_val, label_train, label_val = train_test_split(input_1d_data,
#                                                                                             input_img_data,
#                                                                                             image_labels,
#                                                                                             test_size=0.3,
#                                                                                             shuffle=True,
#                                                                                             random_state=121)
# # 互粉训练集、验证集、测试集=5:3:2
signal_train_val, signal_test, image_train_val, image_test, label_train_val, label_test = train_test_split(input_1d_data,
                                                                                                           input_img_data,
                                                                                                           image_labels,
                                                                                                           test_size=0.2,
                                                                                                           shuffle=True,
                                                                                                           random_state=121)
signal_train, signal_val, image_train, image_val, label_train, label_val = train_test_split(signal_train_val,
                                                                                            image_train_val,
                                                                                            label_train_val,
                                                                                            test_size=0.375,  # 0.375 25
                                                                                            shuffle=True,
                                                                                            random_state=121)
# 查看划分数目
# train = pd.DataFrame(data=label_train)
# train.to_csv('F:/1204/Compare/result/signal_train.csv', index=None, columns=None, header=None, mode='a')
# val = pd.DataFrame(data=label_val)
# val.to_csv('F:/1204/Compare/result/signal_val.csv', index=None, columns=None, header=None, mode='a')
# test = pd.DataFrame(data=label_test)
# test.to_csv('F:/1204/Compare/result/signal_test.csv', index=None, columns=None, header=None, mode='a')




# # ---- 所有图---
# image_train_val, image_test, label_train_val, label_test = train_test_split(input_img_data,
#                                                                             image_labels,
#                                                                             test_size=0.2,
#                                                                             shuffle=True,
#                                                                             random_state=111)
# image_train, image_val, label_train, label_val = train_test_split(image_train_val,
#                                                                   label_train_val,
#                                                                   test_size=0.375,  # 0.375 25
#                                                                   shuffle=True,
#                                                                   random_state=111)

# print('signal_train:-----------', signal_train.shape)  # 1950,2000
#
# print('signal_val:____________', signal_val.shape)  # 1170,2000
#
# print('signal_test:____________', signal_test.shape)  # 780,2000
# data2 = pd.DataFrame(data=signal_test)
# data2.to_csv('F:/1204/Compare/result/DM/Fusion/signal_test1.csv', index=None, columns=None, header=None, mode='a')
#
# print('image_train:_____________', image_train.shape)  # 1950,224,224,3
# # # print('image_train:_____________', image_train)
#
# print('image_val:_____________', image_val.shape)  # 1170,224,224,3
# # # print('image_val:_____________', image_val)
#
# print('image_test:______________', image_test.shape)  # 780,224,224，3
# # # print('image_test:______________', image_test)
# # data2 = pd.DataFrame(data=image_test)
# # data2.to_csv('F:/1204/Compare/result/DM/Fusion/image_test.csv', index=None, columns=None, header=None, mode='a')
#
# print('label_train:_____________', label_train.shape)  # 1950，1
# # print('one_label_train:_____________', label_train)
#
# print('label_val:______________', label_val.shape)  # 1170
# # print('one_label_val:______________', label_val)
#
# print('label_test:______________', label_test.shape)  # 780
# # print('one_label_test:______________', label_test)
# data2 = pd.DataFrame(data=label_test)
# data2.to_csv('F:/1204/Compare/result/DM/Fusion/label_test.csv', index=None, columns=None, header=None, mode='a')


# 创建模型
# input_shape1 = (26, 1)
input_shape1 = (2000, 1)
input_shape2 = (224, 224, 3)
# input_shape2 = (64, 64, 3)
