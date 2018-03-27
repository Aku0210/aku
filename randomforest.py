# ----------------------------------------------------------------------------- #
# ---------------------------- AKSHAY KOTHARI --------------------------------- #
# ----------------------------------------------------------------------------- #


%matplotlib inline
print(__doc__)

# Import all required libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd

# defining Random Forest Classifier
def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

# import some data to play with
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
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, random_state=0)

# Run the classifier on training data
trained_model = random_forest_classifier(X_train1, y_train1)
y_pred1 = trained_model.fit(X_train1, y_train1).predict(X_test1)
y_score = trained_model.fit(X_train1, y_train1).predict_proba(X_test1)

# Code to write the output to csv file
#outfile = open('pred1.csv', 'w')
#out = csv.writer(outfile)
#out.writerows(map(lambda x: [x], y_pred1))
#outfile.close()

# Cross-validation using 10-folds 
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cvs1 = cross_val_score(trained_model, X, y, cv=cv, scoring='f1_macro')

# Using different metrics such as Accuracy, Correlation coefficient and RMSE
MSE = metrics.mean_squared_error(y_test1, y_pred1)
RMSE = np.sqrt(MSE)
Cor_coeff = np.corrcoef(y_test1, y_pred1)
Accuracy = metrics.accuracy_score(y_test1, y_pred1)
train_score = trained_model.score(X_train1, y_train1)
trained_model.fit(X_test1, y_test1)
test_score = trained_model.score(X_test1, y_test1)

# Printing all the metrics
print("MSE: ",MSE)
print("RMSE: ",RMSE)
print("Cor_coeff: ",Cor_coeff)
print("Accuracy: ",Accuracy)
print("train_score: ",train_score)
print("test_score: ",test_score)
print("CVS:",cvs1)

# Code to plot confusion matrix for the above test data
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
    #plt.figure(figsize=(8,8))
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

cnf_matrix = confusion_matrix(y_test, y_pred1)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.show()