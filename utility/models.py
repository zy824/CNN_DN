import tensorflow as tf
from keras import Input
from tensorflow import keras
from keras.regularizers import L2
from tensorflow.keras.models import Model
from utility.args import classnum, input_shape1, input_shape2
from keras.layers import Flatten, Dense, Conv1D, Dropout, Activation, BatchNormalization, MaxPooling1D, LSTM, \
    Bidirectional

'''-------------一、定义BottleNeck模块-----------------------------'''


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(3, 1),
                                            strides=1,
                                            padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(1, 3),
                                            strides=1,
                                            padding="same")
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x, training=training)
        return x


'''-------------二、定义Dense Block模块-----------------------------'''


# BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []
        self.bottle_necks = []
        for i in range(self.num_layers):
            self.bottle_necks.append(BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate))

    def call(self, inputs, training=None, **kwargs):
        self.features_list.append(inputs)
        x = inputs
        for i in range(self.num_layers):
            y = self.bottle_necks[i](x, training=training)
            self.features_list.append(y)
            x = tf.concat(self.features_list, axis=-1)
        self.features_list.clear()
        return x


'''-------------三、构造Transition层-----------------------------'''


# BN+1×1Conv+2×2AveragePooling


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same")
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                              strides=2,
                                              padding="same")
        # self.dp = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        # x = self.dp(x, training=training)
        return x


'''-------------四、搭建DenseNet网络-----------------------------'''


class DenseNet(tf.keras.Model):
    # compression_rate:压缩率
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()

        '''-------------一、构造初始卷积层-----------------------------'''
        # self.conv = tf.keras.layers.Conv2D(filters=num_init_features,
        #                                    kernel_size=(7, 7),
        #                                    strides=2,
        #                                    input_shape=input_shape2,  # ,
        #                                    padding="same")
        # self.bn = tf.keras.layers.BatchNormalization()
        # self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
        #                                       strides=2,
        #                                       padding="same")

        # # 修改3个
        self.conv1 = tf.keras.layers.Conv2D(filters=num_init_features,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.conv2 = tf.keras.layers.Conv2D(filters=num_init_features,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.conv3 = tf.keras.layers.Conv2D(filters=num_init_features,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        # 第一次执行特征的维度来自于前面的特征提取
        self.num_channels = num_init_features

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,32,4）
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels

        # self.droput1 = tf.keras.layers.Dropout(drop_rate)

        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,32,4）
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels

        # self.droput2 = tf.keras.layers.Dropout(drop_rate)
        #
        # # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))
        #
        # # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,32,4）
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels

        # self.droput3 = tf.keras.layers.Dropout(drop_rate)

        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        # x = self.conv(inputs)
        # x = tf.nn.relu(x)
        # x = self.bn(x, training=training)
        # x = self.pool(x)

        # 把bnrelu换成relubn
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = tf.nn.relu(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.dense_block_1(x, training=training)
        # x = self.droput1(x, training=training)
        x = self.transition_1(x, training=training)

        x = self.dense_block_2(x, training=training)
        # x = self.droput2(x, training=training)
        x = self.transition_2(x, training=training)

        x = self.dense_block_3(x, training=training)
        # x = self.droput3(x, training=training)
        x = self.transition_3(x, training=training)

        x = self.dense_block_4(x, training=training)

        x = self.avgpool(x)

        return x


'''--------------创建1DCNN----------------------------------------'''


class CNN_1D(tf.keras.Model):
    def __init__(self):
        super(CNN_1D, self).__init__()
        # self.c1A = Conv1D(64, 3, input_shape=input_1d, padding='same')  # 特征数据
        # self.c1A = Conv1D(64, 3, input_shape=input_shape1, padding='same')
        self.c1A = Conv1D(64, 3, input_shape=input_shape1, padding='same')
        self.a1A = Activation('relu')
        self.c1B = Conv1D(64, 3, padding='same')
        self.a1B = Activation('relu')
        # self.b1B = BatchNormalization()
        self.p1 = MaxPooling1D(2)

        self.c2A = Conv1D(128, 3, padding='same')
        self.a2A = Activation('relu')
        self.c2B = Conv1D(128, 3, padding='same')
        self.a2B = Activation('relu')
        # self.b2B = BatchNormalization()
        self.p2 = MaxPooling1D(2)

        self.c3A = Conv1D(256, 3, padding='same')
        self.a3A = Activation('relu')
        self.c3B = Conv1D(256, 3, padding='same')
        self.a3B = Activation('relu')
        # self.cc3B = Conv1D(256, 3, padding='same')
        # self.aa3B = Activation('relu')
        # # self.b3B = BatchNormalization()
        self.p3 = MaxPooling1D(2)

        # self.lstm0 = LSTM(128, return_sequences=True)
        # # self.lstm1 = LSTM(128, return_sequences=True)
        # self.lstm2 = LSTM(256, return_sequences=True)
        # self.lstm3 = LSTM(256, return_sequences=True)
        # self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.c4A = Conv1D(512, 3, padding='same')
        # self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.a4A = Activation('relu')
        # self.c4B = Conv1D(512, 3, padding='same')
        # self.a4B = Activation('relu')
        # # self.b4B = BatchNormalization()
        # self.p4 = MaxPooling1D(2)

        self.f1 = Flatten()
        # self.d0 = Dense(256)
        # self.r0 = Activation('relu')

        self.d1 = Dense(128)
        self.r1 = Activation('relu')

        self.d2 = Dense(64)
        self.r2 = Activation('relu')

        self.d3 = Dense(32)
        self.r3 = Activation('relu')

        self.d4 = Dense(16)
        self.r4 = Activation('relu')

        self.d5 = Dense(8)
        self.r5 = Activation('relu')

        # self.c5A = Conv1D(512, 3, padding='same')
        # self.a5A = Activation('relu')
        # self.c5B = Conv1D(512, 3, padding='same')
        # self.a5B = Activation('relu')
        # # self.b5B = BatchNormalization()
        # self.p5 = MaxPooling1D(2)

        # self.f1 = Flatten()
        # self.d1 = Dense(1024)
        # self.a6 = Activation('relu', name='OutputF')

        # self.d2 = Dense(classnum)
        # self.a7 = Activation('softmax')

    def call(self, x):
        x = self.c1A(x)
        x = self.a1A(x)
        x = self.c1B(x)
        x = self.a1B(x)
        # x = self.b1B(x)
        x = self.p1(x)

        x = self.c2A(x)
        x = self.a2A(x)
        x = self.c2B(x)
        x = self.a2B(x)
        # x = self.b2B(x)
        x = self.p2(x)

        x = self.c3A(x)
        x = self.a3A(x)
        x = self.c3B(x)
        x = self.a3B(x)
        # # x = self.cc3B(x)
        # # x = self.aa3B(x)
        # # # x = self.b3B(x)
        x = self.p3(x)

        # x = self.lstm0(x)
        # x = self.lstm1(x)
        # x = self.lstm2(x)
        # x = self.lstm3(x)
        # x = self.ln(x)
        # x = self.drp(x)

        x = self.f1(x)
        # x = self.d0(x)
        # x = self.r0(x)
        # x = self.d1(x)
        # x = self.r1(x)
        x = self.d2(x)
        x = self.r2(x)
        x = self.d3(x)
        x = self.r3(x)
        x = self.d4(x)
        x = self.r4(x)
        # x = self.d5(x)
        # x = self.r5(x)

        #
        # x = self.c4A(x)
        # x = self.a4A(x)
        # x = self.c4B(x)
        # x = self.a4B(x)
        # # x = self.b4B(x)
        # x = self.p4(x)

        # x = self.c5A(x)
        # x = self.a5A(x)
        # x = self.c5B(x)
        # x = self.a5B(x)
        # # x = self.b5B(x)
        # x = self.p5(x)

        # x = self.f1(x)
        # x = self.d1(x)
        # x = self.a6(x)

        # x = self.d2(x)
        # outputs = self.a7(x)

        return x  # , a


def densenet():
    return DenseNet(num_init_features=32,
                    growth_rate=16,
                    # num_init_features=64,
                    # growth_rate=32,
                    block_layers=[6, 8, 8, 8],  # 11, 5, 3   2, 3, 4, 2   6, 8, 8, 8  6, 12, 24, 16  3, 6, 12, 8
                    compression_rate=0.5,
                    drop_rate=0.25)



def cnn_1d():
    return CNN_1D()


def fusion():  # input_shape1, input_shape2

    input_1d = Input(shape=input_shape1)
    cnn_net = cnn_1d()(input_1d)
    d_11 = Dense(4, activation='relu', kernel_regularizer=L2(0.01), name='CnnOutput')(cnn_net)

    input_2d = Input(input_shape2)
    densenet_net = densenet()(input_2d)
    d_22 = Dense(4, activation='relu', kernel_regularizer=L2(0.01), name='DenseNetOutput')(densenet_net)

    # 获取一维卷积模型和二维卷积模型的倒数第二层特征
    # 合并两个输入
    merged = tf.keras.layers.concatenate([d_11, d_22], axis=1)
    lb = tf.keras.layers.LayerNormalization(epsilon=1e-6)(merged)

    d_3 = tf.keras.layers.Dense(8, activation='relu')(lb)

    # d_4 = tf.keras.layers.Dense()

    output = tf.keras.layers.Dense(classnum, activation='softmax')(d_3)

    # 创建模型
    model = tf.keras.Model(inputs=[input_1d, input_2d], outputs=output)

    # 模型结构
    print(model.summary())
    # 保存模型结构图
    model_img_file = 'F:/1204/Compare/result/DM/Fusion/model_fusion.png'
    tf.keras.utils.plot_model(model, to_file=model_img_file,
                              show_shapes=True,
                              show_layer_activations=True,
                              show_dtype=True,
                              show_layer_names=True)

    return model










