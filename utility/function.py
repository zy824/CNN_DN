import os
import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
# from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from utility.args import INIT_LR
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, auc, ConfusionMatrixDisplay
from utility.args import batch_size, EPOCHS, classnum
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


# 学习调度函数
def buil_lrfn(lr_star=INIT_LR,
              lr_max=0.00005,
              lr_min=1e-6,
              lr_rampup_epochs=5,  # 前五轮增长，较大
              lr_sustain_epochs=0,  # 保持不变
              lr_exp_decay=0.5):  # 衰减因子

    def lrfn(epoch):
        # 学习率增长阶段
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_star) / lr_rampup_epochs * epoch + lr_star
        # 学习率保持阶段
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        # 学习率衰减阶段
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr

    return lrfn


# 模型训练
def model_train(model, saved_path, x_train, y_train, x_val, y_val):
    # 定义回调函数，学习率调度
    lrfn = buil_lrfn()
    lr_schedule = LearningRateScheduler(lrfn, verbose=1)  # =1表示学习记录有变化会提示

    reduce = ReduceLROnPlateau(monitor='val_accuracy',
                               patience=5,  # 当10个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                               verbose=1,
                               factor=0.5,  # 每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                               min_lr=1e-6)  # 学习率下限

    # 定义回调函数，提前终止训练
    early_stop = EarlyStopping(monitor='val_accuracy',  # 验证集准确率
                               min_delta=0,  # 不上升
                               patience=20,  # 10代
                               verbose=1,
                               restore_best_weights=True)  # 模型回滚到最优模型权重
    # 定义回调函数，保存最优模型
    best_model = ModelCheckpoint(saved_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,  # 只保留最优模型
                                 save_weights_only=True,  # 权重结构都保存
                                 mode='max')  # 保存验证集准确率最高的模型

    history = model.fit(x_train,
                        y_train,
                        # steps_per_epoch=x_train.shape[0] / batch_size,
                        batch_size=batch_size,
                        epochs=EPOCHS,
                        verbose=1,
                        # validation_steps=x_val.shape[0] / batch_size,
                        callbacks=[best_model, lr_schedule],  # , reduce
                        validation_data=(x_val, y_val))

    return history


def plot_confusion_matrix(ccc, ddd, cm, classes, title='Confusion matrix', cmap=plt.cm.jet):  # c, d,
    """

    :param ccc: 光纤类型
    :param ddd: 数据类型
    :param cm:
    :param classes:
    :param title:
    :param cmap:
    :return:
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    if ccc == 1:  # 挂网事件
        plt.xticks(tick_marks, ('background noise', 'panpa', 'paowu', 'huangdong'))
        plt.yticks(tick_marks, ('background noise', 'panpa', 'paowu', 'huangdong'))
    else:  # 地埋事件
        plt.xticks(tick_marks, ('background noise', 'manual digging', 'personnel walking', 'stone throwing'))
        plt.yticks(tick_marks, ('background noise', 'manual digging', 'personnel walking', 'stone throwing'))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predict Label')

    if ccc == 1:  # 挂网
        if ddd == 1:
            plt.savefig('F:/1204/Compare/result/GW/1D/HX-GW-1D.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        elif ddd == 2:
            plt.savefig('F:/1204/Compare/result/GW/2D/HX-GW-2D.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/GW/Fusion/HX-GW-Fusion.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)

    else:  # 地埋
        if ddd == 1:
            plt.savefig('F:/1204/Compare/result/DM/1D/HX-DM-1D.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        elif ddd == 2:
            plt.savefig('F:/1204/Compare/result/DM/2D/HX-DM-2D.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/DM/Fusion/HX-Dm-Fusion.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)

    # 混淆矩阵
    # prediction = model.predict(X_test, verbose=1)
    # predict_label = np.argmax(prediction, axis=1)
    # true_label = np.argmax(y_test, axis=1)
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # cm = confusion_matrix(y_true=true_label, y_pred=predict_label)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()

    plt.show()


# 显示混淆矩阵
def plot_confuse(cc, dd, model, x_val, y_val):  # , c, d
    predictions = model.predict(x_val)
    predictions = np.argmax(predictions, axis=1)
    truelabel = y_val  # .argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(cc, dd, conf_mat, range(int(np.max(truelabel)) + 1))  #


# 9.验证集预测
def val_predict(model, x_val, y_val):
    score = model.evaluate(x_val, y_val)
    print('验证集准确率：', score[1])


# 数据预测
def pred(model, val):
    y_pred = model.predict(val)
    return y_pred


# 特征可视化
def plot_tsne(cc, dd, features, labels, epoch, fileNameDir=None):
    """
    # features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    # label:(N) 有N个标签
    """

    # 创建目标文件夹
    if not os.path.exists(fileNameDir):
        os.makedirs(fileNameDir)
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    # 查看标签的种类有几个
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4

    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)

    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]

    # 颜色是根据标签的大小顺序进行赋色.
    hex = ["#5784db", "#c957db", "#dd5f57", "#b9db57"]  # 绿、红,, "#57db30"'purple', 'red', 'blue', 'green'
    data_label = []
    for v in df.y.tolist():

        if cc == 1:  # 挂网光纤
            if v == 0:
                data_label.append("背景噪声")
            elif v == 1:
                data_label.append("攀爬栅栏")
            elif v == 2:
                data_label.append("栅栏抛物")
            elif v == 3:
                data_label.append("晃动栅栏")

        else:  # 地埋光缆
            if v == 0:
                data_label.append("background noise")
            elif v == 1:
                data_label.append("manual digging")
            elif v == 2:
                data_label.append("personnel walking")
            elif v == 3:
                data_label.append("stone throwing")

    df["value"] = data_label

    # hue=df.y.tolist()
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    # s:指定显示形状的大小
    sns.scatterplot(x=df.comp1.tolist(),
                    y=df.comp2.tolist(),
                    hue=df.value.tolist(),
                    style=df.value.tolist(),
                    palette=sns.color_palette(hex, class_num),
                    markers={"背景噪声": ".",
                             "攀爬栅栏": ".",
                             "栅栏抛物": ".",
                             "晃动栅栏": "."} if cc == 1 else {"background noise": ".",
                                                               "manual digging": ".",
                                                               "personnel walking": ".",
                                                               "stone throwing": "."},
                    # s = 10,
                    data=df).set(title="特征可视化")  # T-SNE projection

    # 指定图注的位置 "lower right"
    # plt.legend(loc="lower right")
    # 不要坐标轴
    # plt_sne.axis("off")
    # 保存图像
    if cc == 1:
        if dd == 1:
            plt.savefig('F:/1204/Compare/result/GW/1D/KSH-GW-vgg.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)
        elif dd == 2:
            plt.savefig('F:/1204/Compare/result/GW/2D/KSH-GW-densenet.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/GW/Fusion/KSH-GW-fusion.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)

    else:
        if dd == 1:
            plt.savefig('F:/1204/Compare/result/DM/1D/KSH-DM-vgg.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)
        elif dd == 2:
            plt.savefig('F:/1204/Compare/result/DM/2D/KSH-DM-densenet.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/DM/Fusion/KSH-DM-fusion.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        transparent=False)

    plt.show()


# 模型评估
def model_estimate(cc, dd, history):
    print("Now,we start drawing the loss and acc trends graph...")
    plt.figure(constrained_layout=True)

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot(2, 1, 1)
    plt.grid(True)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    if cc == 1:
        if dd == 1:
            plt.savefig('F:/1204/Compare/result/GW/1D/GW_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/GW/1D/GW_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/GW/1D/GW_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/GW/1D/GW_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/GW/1D/GW_val_loss.npy', val_loss)
        elif dd == 2:
            plt.savefig('F:/1204/Compare/result/GW/2D/GW_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/GW/2D/GW_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/GW/2D/GW_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/GW/2D/GW_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/GW/2D/GW_val_loss.npy', val_loss)
        else:
            plt.savefig('F:/1204/Compare/result/GW/Fusion/GW_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/GW/Fusion/GW_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/GW/Fusion/GW_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/GW/Fusion/GW_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/GW/Fusion/GW_val_loss.npy', val_loss)

    else:
        if dd == 1:
            plt.savefig('F:/1204/Compare/result/DM/1D/DM_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/DM/1D/DM_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/DM/1D/DM_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/DM/1D/DM_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/DM/1D/DM_val_loss.npy', val_loss)
        elif dd == 2:
            plt.savefig('F:/1204/Compare/result/DM/2D/DM_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/DM/2D/DM_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/DM/2D/DM_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/DM/2D/DM_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/DM/2D/DM_val_loss.npy', val_loss)
        else:
            plt.savefig('F:/1204/Compare/result/DM/Fusion/DM_acc_loss.jpg')
            acc_num = np.save('F:/1204/Compare/result/DM/Fusion/DM_acc.npy', acc)
            val_acc_num = np.save('F:/1204/Compare/result/DM/Fusion/DM_val_acc.npy', val_acc)
            loss_num = np.save('F:/1204/Compare/result/DM/Fusion/DM_loss.npy', loss)
            val_loss_num = np.save('F:/1204/Compare/result/DM/Fusion/DM_val_loss.npy', val_loss)

    plt.show()


# 测试集测试
def test_predict(cc, dd, model, path, image_test, label_test):
    # 真实值与预测值

    model.load_weights(path)
    t1 = time.time()
    print('t1', t1)

    y_pred = model.predict(image_test)
    y_pred = [np.argmax(i) for i in y_pred]
    y_pred = np.array(y_pred)
    # print(y_pred.shape)
    # print(y_pred)
    # data1 = pd.DataFrame(data=y_pred)
    # data1.to_csv('F:/1204/Compare/result/DM/2D/test_pred.csv', index=None, columns=None, header=None, mode='a')

    y_true = label_test
    # y_true = [np.argmax(i) for i in y_true]
    y_true = np.array(y_true)
    # print(y_true.shape)
    # print(y_true)
    # data2 = pd.DataFrame(data=y_true)
    # data2.to_csv('F:/1204/Compare/result/DM/2D/test_true.csv', index=None, columns=None, header=None, mode='a')

    # 测试集混淆矩阵
    # prediction = model.predict(image_test, verbose=1)
    # predict_label = np.argmax(prediction, axis=1)
    # true_label = np.argmax(label_test, axis=1)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if cc == 1:
        if dd == 2:
            plt.savefig('F:/1204/Compare/result/GW/2D/HX-GW-2D-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        elif dd == 1:
            plt.savefig('F:/1204/Compare/result/GW/1D/HX-GW-1D-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/GW/Fusion/HX-GW-Fusion-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
    else:
        if dd == 2:
            plt.savefig('F:/1204/Compare/result/DM/2D/HX-DM-2D-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        elif dd == 1:
            plt.savefig('F:/1204/Compare/result/DM/1D/HX-DM-1D-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)
        else:
            plt.savefig('F:/1204/Compare/result/DM/Fusion/HX-DM-Fusion-test.jpg',
                        dpi=200,
                        bbox_inches='tight',
                        ransparent=False)


    plt.show()

    accuracy_avg = accuracy_score(y_true, y_pred)
    print(f"测试集准确率: {accuracy_avg}")

    t2 = time.time()
    print('t2', t2)
    t3 = t2 - t1
    print('time=', t3)

    # ------------------------------精确率，召回率，F1分数----------------------------------------------------------------
    # micro：计算全局的指标
    # macro：对每个类别独立计算指标并求平均
    # weighted：对每个类别独立计算指标并按照它们的实例数量加权求平均
    # None：返回每个类别的指标，不进行平均
    accuracy_score_value = accuracy_score(label_test, y_pred)
    recall_score_value = recall_score(y_true, y_pred, average='weighted')
    precision_score_value = precision_score(y_true, y_pred, average='weighted')
    f1score = f1_score(y_true, y_pred, average='weighted')
    classification_report_value = classification_report(label_test, y_pred)
    print("准确率：", accuracy_score_value)
    print("召回率：", recall_score_value)
    print("精确率：", precision_score_value)
    print("f1-Score：", f1score)
    print("Report : ", classification_report_value)

    # plt.figure(figsize=(10, 8))
    # tsne = TSNE(n_components=2).fit_transform(y_true)
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=label_test)
    # plt.colorbar()
    # plt.show()

    """# 分类标签的一次热编码
    def one_hot_encoding(data):
        L_E = LabelEncoder()
        integer_encoded = L_E.fit_transform(data)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded_data = onehot_encoder.fit_transform(integer_encoded)
        return one_hot_encoded_data

    Y_Test = one_hot_encoding(y_true.ravel())
    print('Y_Test', Y_Test.shape)
    print('y_pred', y_pred.shape)

    def plot_multiclass_roc(Y_Test, Predictions):
        # Compute ROC curve and Area Under Curve (AUC) for each class

        Y_Test = Y_Test.reshape(-1, 1)
        # print(Y_Test.shape)  # 3120
        Predictions = Predictions.reshape(-1, 1)
        # print(Predictions.shape)  # 780

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(classnum):
            fpr[i], tpr[i], _ = roc_curve(Y_Test[:, i], Predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_Test.ravel(), Predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classnum)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(classnum):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= classnum

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(20, 10))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        fpr5 = np.save(r'F:/1204/Compare/result/DM/1D//fpr_micro.npy', fpr["micro"], allow_pickle=True,
                       fix_imports=True)  # 保存成np
        tpr5 = np.save(r'F:/1204/Compare/result/DM/1D/tpr_micro.npy', tpr["micro"], allow_pickle=True,
                       fix_imports=True)  # 保存成np

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})''#'.format(roc_auc["macro"]),
                 ############################################################
                 color='navy', linestyle=':', linewidth=4)
        fpr6 = np.save(r'F:/1204/Compare/result/DM/1D/fpr_macro.npy', fpr["macro"], allow_pickle=True,
                       fix_imports=True)  # 保存成np
        tpr6 = np.save(r'F:/1204/Compare/result/DM/1D/tpr_macro.npy', tpr["macro"], allow_pickle=True,
                       fix_imports=True)  # 保存成np

        for i in range(classnum):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})''#'.format(i, roc_auc[
                         i]))  # #############################################################删#
            fp = np.save('F:/1204/Compare/result/DM/1D/ftpr1/fpr_' + '%d.npy' % i, fpr[i], allow_pickle=True,
                         fix_imports=True)  # 保存成np
            tp = np.save('F:/1204/Compare/result/DM/1D/ftpr1/tpr_' + '%d.npy' % i, tpr[i], allow_pickle=True,
                         fix_imports=True)  # 保存成np

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('MultiClass ROC Plot with Respective AUC', fontsize=25)
        plt.legend(loc="lower right")
        plt.show()

    plot_multiclass_roc(Y_Test, y_pred)
"""

    """# 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classnum):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classnum)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(classnum):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= classnum
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(classnum), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2).fit_transform(y_true)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=label_test)
    plt.colorbar()
    plt.show()"""
