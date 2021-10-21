import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def fairness_unawareness(args, x_train, x_test, y_train, y_test):
    print("run classifier")
    print(x_train.shape)
    print(y_train.shape)
    classifier = svm.LinearSVC(multi_class="crammer_singer")
    classifier.fit(x_train, y_train.ravel())
    print("print confusion matrix")
    predictions = classifier.predict(x_test)
    cm = confusion_matrix(y_test, predictions, labels=classifier.classes_, normalize='true')
    display = ConfusionMatrixDisplay(cm)
    print("Confusion matrix results")
    print(display.confusion_matrix)
    display.plot()
    plt.show()
    
    plt.savefig("./output/fairness_unawareness_res.png")
    plt.close()
