def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./data/Human_Activity_Recognition_Using_Smartphones_Data_V2.csv')

print(data.shape)

float_columns = (data.dtypes == np.float64)

print('\n')
print(float_columns)


le = LabelEncoder()

data['Activity'] = le.fit_transform(data['Activity'])

print('\n\n')
print(le.classes_)

feature_columns = [x for x in data.columns if x!= 'Activity']

x_train, x_test, y_train, y_test = train_test_split(data[feature_columns], data['Activity'], test_size=0.3, random_state=42)

print('\n\n')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


'''

error_list = list()

tree_list = [15,25,50,100,200,400]

for n_trees in tree_list:
    GBC = GradientBoostingClassifier(max_features = 5, n_estimators=n_trees, random_state=42)
    GBC.fit(x_train.values , y_train.values)
    y_pred = GBC.predict(x_test)

    error = 1.0 - accuracy_score(y_test,y_pred)

    error_list.append(pd.Series({'n_trees':n_trees, 'error': error}))

error_df = pd.concat(error_list, axis=1).T.set_index('n_trees')

print('\n\n')
print(error_df)

'''

'''
param_grid = {'learning_rate':[0.1,0.01,0.001],
              'subsample':[1.0,0.5],
              'max_features':[2,3,4],
              'n_estimators':[15,25,50,100,200,400]}

GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42),param_grid=param_grid,scoring='accuracy',n_jobs=-1)

GV_GBC = GV_GBC.fit(x_train,y_train)

print(GV_GBC.best_estimator_) 

pickle.dump(GV_GBC,open('gv_gbc.p','wb'))
'''

GV_GBC = pickle.load(open('gv_gbc.p','rb'))
y_pred = GV_GBC.predict(x_test)
print('\n\n')
print(classification_report(y_pred, y_test))

'''
ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))

param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.001]}

GV_ABC = GridSearchCV(ABC,
                      param_grid=param_grid, 
                      scoring='accuracy',
                      n_jobs=-1)

GV_ABC = GV_ABC.fit(x_train, y_train)

pickle.dump(GV_ABC,open('gv_abc.model','wb'))
'''

GV_ABC = pickle.load(open('gv_abc.model','rb'))
y_pred = GV_ABC.predict(x_test)
print(classification_report(y_pred, y_test))

LR_L2 = LogisticRegression(penalty='l2', max_iter=500, solver='saga').fit(x_train, y_train)

y_pred = LR_L2.predict(x_test)
print(classification_report(y_pred, y_test))

estimators = [('LR_L2', LR_L2), ('GBC', GV_GBC)]

# Though it wasn't done here, it is often desirable to train 
# this model using an additional hold-out data set and/or with cross validation
VC = VotingClassifier(estimators, voting='soft')
VC = VC.fit(x_train, y_train)
y_pred = VC.predict(x_test)
print(classification_report(y_test, y_pred))


sns.set_context('talk')
cm = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cm, annot=True, fmt='d')
plt.show()
