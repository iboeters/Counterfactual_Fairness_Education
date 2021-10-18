import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

def fairness_unaware(args, x_train, x_test, y_train, y_test):
	classifier = svm.SCV(kernel='linear').fit(x_train, y_train)
	
	display = ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test, cmap=plt.cm.Blues, normalize=True)
	display.ax_.set_title("Confusion matrix results")
	print("Confusion matrix results")
	print(display.confusion_matrix)
	plt.show()

	plt.savefig("./output/fairness_unawareness_res.png")
	plt.close()
