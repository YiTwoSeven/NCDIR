import itertools
import matplotlib.pyplot as plt
import numpy as np

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("222.png", dpi=300)


cnf_matrix = np.array([[400, 0, 1, 3, 9, 62, 3, 0],
                       [0, 69, 7, 0, 2, 18, 2, 1],
                       [1, 0, 20, 2, 0, 2, 1, 0],
                       [0, 1, 0, 94, 0, 0, 4, 0],
                       [1, 1, 0, 0, 144, 1, 1, 0],
                       [65, 6, 3, 0, 14, 519, 6, 0],
                       [1, 3, 0, 1, 6, 3, 95, 0],
                       [0, 0, 0, 0, 0, 0, 0, 25]])

# cnf_matrix = np.array([[121, 4, 60, 0, 27, 29, 121, 117],
#                       [0, 48, 1, 36, 0, 8, 2, 3],
#                       [1, 1, 5, 1, 1, 14, 2, 1],
#                       [1, 26, 1, 64, 0, 6, 1, 0],
#                       [3, 15, 6 ,1, 95, 26, 2, 0], 
#                       [51, 52, 120, 6, 30, 160, 110, 84],
#                       [2, 24, 5, 13, 13, 39, 11, 2],
#                       [0, 24, 0, 1, 0, 0, 0, 0]])
attack_types = ['0', '1', '2', '3', '4', '5', '6', '7']
plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')
