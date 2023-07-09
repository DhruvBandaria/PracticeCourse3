import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,LabelBinarizer,OrdinalEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


df = pd.read_csv('./data/churndata_processed.csv')
#df = churndata.drop(columns=['id','phone','total_revenue','cltv','churn_score'],axis=1)
print(round(df.describe(),2))

df_uniques = df.nunique()
print('\n\n')
print(df_uniques)

binary_variables = list(df_uniques[df_uniques == 2].index)
print('\n\n')
print(binary_variables)

categorical_variables = list(df_uniques[(df_uniques > 2) & (df_uniques <=6)].index)
print('\n\n')
print(categorical_variables)

print('\n\n')
print([[i, list(df[i].unique())] for i in categorical_variables])

ordinal_variables = ['contract','satisfaction']
print('\n\n')
print(df['months'].unique())

ordinal_variables.append('months')

numeric_variables = list(set(df.columns) - set(ordinal_variables) - set(categorical_variables) - set(binary_variables))

df['months'] = pd.cut(df['months'], bins=5)

lb , le = LabelBinarizer(), LabelEncoder()

for column in ordinal_variables:
    df[column] = le.fit_transform(df[column])

for column in binary_variables:
    df[column] = lb.fit_transform(df[column])

categorical_variables = list(set(categorical_variables) - set(ordinal_variables) - set(binary_variables))

df = pd.get_dummies(df, columns= categorical_variables, drop_first=True)

print('\n\n')
print(df.describe().T)

mm = MinMaxScaler()

for column in [ordinal_variables + numeric_variables]:
    df[column] = mm.fit_transform(df[column])


print('\n\n')
print(df.describe().T)

y, x = df['churn_value'], df.drop(columns='churn_value')

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print('\n\nK=3')
print(classification_report(y_test,y_pred))
print('Accuracy Score',round(accuracy_score(y_test,y_pred),2))
print('F1 Score',round(f1_score(y_test,y_pred),2))


knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn = knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print('\n\nK=5')
print(classification_report(y_test,y_pred))
print('Accuracy Score',round(accuracy_score(y_test,y_pred),2))
print('F1 Score',round(f1_score(y_test,y_pred),2))

max_k = 40
f1_scores = list()
error_rates = list()

for k in range(1,max_k):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn = knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    f1 = f1_score(y_test,y_pred)
    f1_scores.append((k,round(f1,4)))
    error = 1- round(accuracy_score(y_test,y_pred),4)
    error_rates.append((k,error))

f1_results = pd.DataFrame(f1_scores,columns=['K','F1 Score'])
error_results = pd.DataFrame(error_rates,columns=['K','Error Rate'])

print('\n\n')
print(f1_results)

print('\n\n')
print(error_results)
