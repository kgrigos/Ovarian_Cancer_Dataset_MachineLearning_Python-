#Libraries
import numpy as np
import pandas as pd
import scikitplot as skplt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xlsxwriter
#read datast
df = pd.read_excel('C:\\Users\\kosta\\Desktop\\Ovarian\\Data.xlsx')
#load the specific features that the bat algorithm gave us
input= pd.read_excel("C:\\Users\\kosta\\Desktop\\Ovarian\\Ovarian3.xlsx")
X=list(input)
#use those features as X
X=df.iloc[:,X]
print(X)
#use the class column as Y
y = df.loc[:, 'class']
#split data and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#fit for Guassian
Gauss = GaussianNB()
Gaussclf = Gauss.fit(X_train, y_train)
Gauss_predicted = Gaussclf.predict(X_test)
#fit for kmeans
Kmeans=KNeighborsClassifier()
Kmeansclf=Kmeans.fit(X_train, y_train)
Kmeans_predicted = Kmeansclf.predict(X_test)
#fit for decision trr
DesTree=DecisionTreeClassifier()
DesTreeclf=DesTree.fit(X_train, y_train)
DesTree_predicted = DesTreeclf.predict(X_test)
#fit for logistic regression
Logistic=LogisticRegression()
Logisticclf=Logistic.fit(X_train, y_train)
Logistic_predicted = Logisticclf.predict(X_test)
#keep from classification the specific 3 information
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
DesTreetable=classification_report(y_test,DesTree_predicted,output_dict=True)
DesTreeaccuracy=DesTreetable['accuracy']
DesTreefscore=DesTreetable['macro avg']['f1-score']
Kmeanstable=classification_report(y_test,Kmeans_predicted,output_dict=True)
Kmeansaccuracy=Kmeanstable['accuracy']
Kmeansfscore=Kmeanstable['macro avg']['f1-score']
Gausstable=classification_report(y_test,Gauss_predicted,output_dict=True)
Gaussaccuracy=Gausstable['accuracy']
Gaussfscore=Gausstable['macro avg']['f1-score']
Logistictable=classification_report(y_test,Logistic_predicted,output_dict=True)
Logisticaccuracy=Logistictable['accuracy']
Logisticfscore=Logistictable['macro avg']['f1-score']
#find probabilities for AUC_SCORE
DesTree_probs=DesTreeclf.predict_proba(X_test)
DesTree_probs=DesTree_probs[:,1]
Kmeans_probs=Kmeansclf.predict_proba(X_test)
Kmeans_probs=Kmeans_probs[:,1]
Gauss_probs=Gaussclf.predict_proba(X_test)
Gauss_probs=Gauss_probs[:,1]
Logistic_probs=Logisticclf.predict_proba(X_test)
Logistic_probs=Logistic_probs[:,1]
from sklearn.metrics import roc_curve,roc_auc_score
DesTreeclf_auc=roc_auc_score(y_test,DesTree_probs)
Kmeansclf_auc=roc_auc_score(y_test,Kmeans_probs)
Gaussclf_auc=roc_auc_score(y_test,Gauss_probs)
Logisticclf_auc=roc_auc_score(y_test,Logistic_probs)
#create for lists for the classifiers name,fscore,Auc score and test set accuracy
names=["Guassian Classifier",'Decision Tree Classifier','Kmeans Classifier','Logistic Regression Classifier']
fscore=[Gaussfscore,DesTreefscore,Kmeansfscore,Logisticfscore]
AUC=[Gaussclf_auc,DesTreeclf_auc,Kmeansclf_auc,Logisticclf_auc]
Accuracy=[Gaussaccuracy,DesTreeaccuracy,Kmeansaccuracy,Logisticaccuracy]
#save the results to an Excel file
import openpyxl
Export={'Name':["Guassian Classifier",'Decision Tree Classifier','Kmeans Classifier','Logistic Regression Classifier'],
        'F1score':[Gaussfscore,DesTreefscore,Kmeansfscore,Logisticfscore],
        'AUC score':[Gaussclf_auc,DesTreeclf_auc,Kmeansclf_auc,Logisticclf_auc],
        'Test_Set Accuracy':[Gaussaccuracy,DesTreeaccuracy,Kmeansaccuracy,Logisticaccuracy]}
DataExport = pd.DataFrame(Export, columns = ['Name', 'F1score','AUC score','Test_Set Accuracy'])
DataExport.to_excel ('C:\\Users\\kosta\\Desktop\\Dataset\\OvarianResults3.xlsx', index = False, header=True)




