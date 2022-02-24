import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

data = pd.read_csv('WineQT.csv')

from sklearn.model_selection import train_test_split
#Logistic Regression
from sklearn.linear_model import LogisticRegression
#Support Vector Machine
from sklearn.svm import SVC
#Naive Bayes (Gaussian, Multinomial)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier
#KNN (k-nearest neighbor)
from sklearn.neighbors import KNeighborsClassifier
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
#XGBoost Classifier
from xgboost.sklearn import XGBClassifier


target = data['quality']
data= data.drop(['quality'],axis=1)


xtrain,xtest,ytrain,ytest = train_test_split(data,target,test_size=0.2,random_state=42)

models = [LogisticRegression(),SVC(),GaussianNB(),MultinomialNB(),SGDClassifier(),KNeighborsClassifier(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]

score={}
for i in models:
    model = i
    model = model.fit(xtrain,ytrain)
    score[str(str(i)[:str(i).index("(")])] = model.score(xtrain,ytrain)

print("best algorithm is",max(score),"with",max(score.values()),"Accuracy")