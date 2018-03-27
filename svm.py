%matplotlib inline
print(__doc__)

# Importing all the required libraries
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd

# Import some data to play with
df = pd.read_csv("day_1.csv")

# Import the columns to be used
X = df.iloc[:,[2]].values
y = df.iloc[:,6].values

# Select the unique class names and binarize them to 0 and 1 
# Here leakage detected is one class and no leakage detected is another class
class_names = np.unique(y)
y1 = label_binarize(y, classes=[0,1])
n_classes = y1.shape[1]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='rbf', gamma=0.003, C=2)
y_pred = classifier.fit(X_train, y_train).predict(X_test)\
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Cross-validation using 10-folds
cv = ShuffleSplit(n_splits=10,test_size=0.3,random_state=0)
cvs = cross_val_score(classifier, X, y, cv=cv, scoring='f1_macro')

# Code to write the output to csv file
#outfile = open('pred.csv', 'w')
#out = csv.writer(outfile)
#out.writerows(map(lambda x: [x], y_pred))
#outfile.close()

# Using different metrics such as RMSE, correlation coefficient and accuracy
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
Cor_coeff = np.corrcoef(y_test, y_pred)
Accuracy = metrics.accuracy_score(y_test, y_pred)
train_score = classifier.score(X_train, y_train)
classifier.fit(X_test, y_test)
test_score = classifier.score(X_test, y_test)

# Printing all the metrics
print("MSE: ",MSE)
print("RMSE: ",RMSE)
print("Cor_coeff: ",Cor_coeff)
print("Accuracy: ",Accuracy)
print("train_score: ",train_score)
print("test_score: ",test_score)
print("Cross validation score:", cvs)

# Code to plot confusion matrix for the above data
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# Code to plot ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(12,8))
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"], linewidth='4')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Ideal Curve',linestyle='--',linewidth='4')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()
plt.show()