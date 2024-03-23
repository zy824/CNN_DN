import os
import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from utility.model import fusion
from tensorflow.keras.optimizers import Adam
from utility.args import c, d, signal_train, label_train, image_train, signal_val, image_val, label_val, signal_test, \
    image_test, label_test, INIT_LR
from utility.function import model_train, plot_confuse, val_predict, model_estimate, pred, plot_tsne, test_predict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用 GPU 0


# 主函数
def main_fusion(cc, dd):
    # 6.模型定义
    model = fusion()
    # model = fusion2()
    # 模型信息
    # model_inf(model)

    model.compile(optimizer=Adam(learning_rate=INIT_LR),
                  loss='sparse_categorical_crossentropy',  # sparse_
                  metrics=['accuracy'])

    # 7.模型训练
    if cc == 1:
        saved_path = "F:/1204/Compare/result/GW/Fusion/Fusion-G/Fusion-G.ckpt"
    else:
        saved_path = "F:/1204/Compare/result/DM/Fusion/Fusion-D/Fusion-D.ckpt"

    if os.path.exists(saved_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(saved_path)

        # x_train = np.expand_dims(signal_train, axis=2)
        # x_val = np.expand_dims(signal_val, axis=2)

    # 训练
    x_train = [signal_train, image_train]
    x_val = [signal_val, image_val]
    x_test = [signal_test, image_test]

    history = model_train(model, saved_path, x_train, label_train, x_val, label_val)

    # 预测
    image_pred = pred(model, x_val)

    # 模型评估,acc-loss图
    model_estimate(cc, dd, history)

    # 显示混淆矩阵
    plot_confuse(cc, dd, history.model, x_val, label_val)

    # 验证集验证
    val_predict(model=model, x_val=x_val, y_val=label_val)
    y_pred_1d = model.predict(x_val)

    # 测试集测试
    test_predict(cc, dd, model, path=saved_path, image_test=x_test, label_test=label_test)

    # # 可视化
    # # 倒数第一层
    # model1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('CnnOutput').output)
    # features = model1(x_val)
    # # print(features)
    # labels = np.argmax(model(x_val), axis=-1)
    # plot_tsne(c, 1, features, labels, "five_visual_cnn", fileNameDir="cnn_feature")
    #
    # # 倒数第一层
    # model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer('DenseNetOutput').output)
    # features = model2(x_val)
    # # print(features)
    # labels = np.argmax(model(x_val), axis=-1)
    # plot_tsne(c, 2, features, labels, "five_visual_dense", fileNameDir="densenet_feature")

    # 倒数第一层
    model3 = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
    features = model3(x_val)
    # print(features)
    labels = np.argmax(model(x_val), axis=-1)
    plot_tsne(c, 3, features, labels, "five_visual_fusion", fileNameDir="fusion_feature")

    # # 最后一层特征可视化
    # model1 = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)  #
    # features = model1(x_val)
    # #     print(features)
    # labels = np.argmax(model(x_val), axis=-1)
    # plot_tsne(c, d, features, labels, "visual_2d", fileNameDir="cnn_lstm_feature")

    return model


if __name__ == '__main__':
    # 调用主函数
    fusion_model = main_fusion(c, d)
