# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *


def clf_decision_tree():
    iris = load_iris()
    data = iris.data
    target = iris.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_test, y_test, cv=10, scoring='accuracy')
    print('平均scores:', scores.mean())
    y_pred = clf.predict(x_test)
    print('y_预测值:', y_pred)
    print('y_实际值:', y_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    print('confusion_matrix:\n', c_matrix)
    import matplotlib.pyplot as plt

    dis = plot_confusion_matrix(clf, x_test, y_test)
    plt.show()
    kappa = cohen_kappa_score(y_test, y_pred)
    print('kappa:', kappa)
    c_report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print('精度报告：\n', c_report)


if __name__ == '__main__':
    clf_decision_tree()
