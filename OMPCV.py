from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.metrics import mean_squared_error
from util import *

def OMPC(x_train,y_train,x_test,y_test):
    reg = OrthogonalMatchingPursuitCV(cv=5).fit(x_train, y_train)
    pred = reg.predict(x_test)
    print("MSE of OMPC: ", mean_squared_error(y_test, pred))
    print(reg.coef_)
    print("Most impacting stock is:",x_train.columns[np.argmax(reg.coef_)])
    plot(pred, y_test,"OMPC")

