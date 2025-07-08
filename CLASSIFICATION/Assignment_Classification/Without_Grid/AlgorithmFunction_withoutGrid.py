import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
            


class Algorithms(): 
         
    def setup(self):            
            dataset=pd.read_csv("CKD.csv")
            dataset
            dataset["classification"].value_counts()
            dataset = pd.get_dummies(dataset, dtype = int)
            indep=dataset[['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv',
                   'wc', 'rc', 'sg_a', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 'rbc_abnormal',
                   'rbc_normal', 'pc_abnormal', 'pc_normal', 'pcc_notpresent',
                   'pcc_present', 'ba_notpresent', 'ba_present', 'htn_no', 'htn_yes',
                   'dm_no', 'dm_yes', 'cad_no', 'cad_yes', 'appet_poor', 'appet_yes',
                   'pe_poor', 'pe_yes', 'ane_no', 'ane_yes']]
            dep=dataset[['classification_no']]
            global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(indep, dep, test_size = 1/3, random_state = 0)
            
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
    
    def SVM(self):
            from sklearn.svm import SVC
            classifier1 = SVC(kernel = 'rbf', random_state = 0)
            classifier1.fit(X_train, y_train)
            y_pred1 = classifier1.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred1)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report1 = classification_report(y_test, y_pred1)
            print(clf_report1)
           
    def RF(self):
            from sklearn.ensemble import RandomForestClassifier
            classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier2.fit(X_train, y_train)
            y_pred2 = classifier2.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred2)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report2 = classification_report(y_test, y_pred2)
            print(clf_report2)
    def DT(self):
            from sklearn.tree import DecisionTreeClassifier
            classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            classifier3.fit(X_train, y_train)
            y_pred3 = classifier3.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred3)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report3 = classification_report(y_test, y_pred3)
            print(clf_report3)
    def KNN(self):
            from sklearn.neighbors import KNeighborsClassifier
            classifier4 = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
            classifier4.fit(X_train, y_train)
            y_pred4 = classifier4.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred4)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report4 = classification_report(y_test, y_pred4)
            print(clf_report4)
    def LGR(self):
            from sklearn.linear_model import LogisticRegression
            classifier5=LogisticRegression(random_state=0)
            classifier5.fit(X_train, y_train)
            y_pred5 = classifier5.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred5)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report5 = classification_report(y_test, y_pred5)
            print(clf_report5)
    def NB(self):
            from sklearn.naive_bayes import GaussianNB
            classifier6=GaussianNB()
            classifier6.fit(X_train, y_train)
            y_pred6 = classifier6.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix
            cm = multilabel_confusion_matrix(y_test, y_pred6)
            print(cm)
            from sklearn.metrics import classification_report
            clf_report6 = classification_report(y_test, y_pred6)
            print(clf_report6)
        
