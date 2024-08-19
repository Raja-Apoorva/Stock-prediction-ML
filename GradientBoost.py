from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
# from util import *

def GB(x_train,y_train,x_test,y_test):
    breg = GradientBoostingClassifier()
    breg.fit(x_train, y_train)
    pred = breg.predict(x_test)
    # print("MSE of Gradient Boosting Regressor: ", mean_squared_error(y_test, pred))
    # plot(pred, y_test,"Gradient Boost Regressor")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, pred))
    # Optional: Print confusion matrix
    print(confusion_matrix(y_test, pred))
