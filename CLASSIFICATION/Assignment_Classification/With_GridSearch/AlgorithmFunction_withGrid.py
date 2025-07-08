import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
            


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
            param_grid = {'kernel':['rbf','poly','sigmoid','linear'],
             'C':[10,100,1000,2000,3000],'gamma':['auto','scale']}
            grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1)
            grid.fit(X_train, y_train)
            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)
            y_pred = grid.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
           
    def RF(self):
            from sklearn.ensemble import RandomForestClassifier
            param_grid = {'criterion':['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt','log2'],
             'n_estimators':[10,100]}
            grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3,n_jobs=-1)
            grid.fit(X_train, y_train)
         
            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)
            y_pred = grid.predict(X_test)
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
    def DT(self):
            from sklearn.tree import DecisionTreeClassifier
            param_grid = {'criterion':['gini', 'entropy', 'log_loss'],
             'max_features': ['sqrt','log2'],
             'splitter':['best','random']}
            grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose
             = 3,n_jobs=-1)
     
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            
            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)

            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            
 
    def KNN(self):
            from sklearn.neighbors import KNeighborsClassifier
            
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'weights': ['uniform', 'distance']}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose
                     = 3,n_jobs=-1)
                    
            grid.fit(X_train, y_train)
                 
            y_pred = grid.predict(X_test)

            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)
           
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            
    def LGR(self):
            from sklearn.linear_model import LogisticRegression
            param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']}
            grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose
                     = 3,n_jobs=-1)
            
                 
           
                    
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)
            
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
                
    def NB(self):
            from sklearn.naive_bayes import GaussianNB
            param_grid = {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                }
            grid = GridSearchCV(GaussianNB(), param_grid, refit = True, verbose
                     = 3,n_jobs=-1)
           
            
                    
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            print("Best Parameters:", grid.best_params_)
            print("Best Cross-Validation Score:", grid.best_score_)
            
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            print(multilabel_confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
        
