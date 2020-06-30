import numpy as np
from model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

def SVM(feat_vec,y):
    f_score = 0.0
    for i in range(10):
        score = 0
        X_train, X_test, y_train, y_test = train_test_split(np.array(feat_vec), y, test_ratio=0.2)
        clf = OneVsRestClassifier(svm.SVC(C=0.1,kernel='rbf',max_iter = 10))
        clf.fit(X_train,y_train)
        for c in range(len(y_test)):
            if  y_test[c] == clf.predict(X_test)[c]:
                score += 1
        f_score += score / len(y_test)
    # print(clf.predict(X_test),y_test)
    return f_score/10