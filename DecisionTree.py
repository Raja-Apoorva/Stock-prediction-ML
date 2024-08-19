from sklearn import tree
from sklearn.metrics import mean_squared_error
# from util import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def DecisionTree(x_train,y_train,x_test,y_test):
    dclf = DecisionTreeClassifier()
    dclf = dclf.fit(x_train, y_train)

    tree.plot_tree(dclf)

    pred = dclf.predict(x_test)
    train_error = np.round(dclf.score(x_train, y_train), 2)
    test_error = np.round(dclf.score(x_test, y_test), 2)
    print(train_error,test_error)
    # print("MSE of Decision tree regressor: ", mean_squared_error(y_test, pred))
    # plot(pred, y_test,"Decision Tree Regressor")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, pred))
    # Optional: Print confusion matrix
    print(confusion_matrix(y_test, pred))

