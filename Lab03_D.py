import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from io import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

from sklearn import tree

data = pd.read_csv('./data/Wine_Quality_Data_V2.csv')
print(data.head())

print('\n\n')
print(data.dtypes)

data['color'] = data.color.replace('white',0).replace('red',1).astype(np.int16)

feature_cols = [x for x in data.columns if x not in 'color']

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)

train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols], data['color']))

x_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'color']

x_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'color']

print('\n\n')
print(y_train.value_counts(normalize=True).sort_index())
print('\n')
print(y_test.value_counts(normalize=True).sort_index())

dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(x_train,y_train)

print('\n\n')
print("Node Count:",dt.tree_.node_count)
print("Max Depth:",dt.tree_.max_depth)

def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy': accuracy_score(y_true, y_pred),
                      'precison': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},name=label)


y_train_pred = dt.predict(x_train)
y_test_pred = dt.predict(x_test)

train_test_full_error = pd.concat([measure_error(y_train,y_train_pred,'train'),
                                   measure_error(y_test, y_test_pred,'test')],axis=1)

print('\n\n')
print(train_test_full_error)

pram_grid = {'max_depth':range(1,dt.tree_.max_depth+1,2),
             'max_features': range(1, len(dt.feature_importances_) + 1)}

GR = GridSearchCV(DecisionTreeClassifier(random_state=42),
                  param_grid=pram_grid,
                  scoring='accuracy',
                  n_jobs=-1)

GR = GR.fit(x_train,y_train)

print('\n\n')
print("GR Node Count:",GR.best_estimator_.tree_.node_count)
print("GR Max Depth:",GR.best_estimator_.tree_.max_depth)

y_train_pred_gr = GR.predict(x_train)
y_test_pred_gr = GR.predict(x_test)

train_test_full_error_gr = pd.concat([measure_error(y_train,y_train_pred_gr,'train'),
                                   measure_error(y_test, y_test_pred_gr,'test')],axis=1)

print('\n\n')
print(train_test_full_error_gr)


tree.plot_tree(GR.best_estimator_,feature_names=feature_cols,filled=True)
plt.savefig('Test.png')


feature_cols = [x for x in data.columns if x not in 'color']
x_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'residual_sugar']

x_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'residual_sugar']

dr = DecisionTreeRegressor().fit(x_train,y_train)
pram_grid = {'max_depth':range(1,dr.tree_.max_depth+1,2),
             'max_features': range(1, len(dr.feature_importances_) + 1)}

GR_sugar = GridSearchCV(DecisionTreeRegressor(random_state=42),
                  param_grid=pram_grid,
                  scoring='neg_mean_squared_error',
                  n_jobs=-1)

GR_sugar = GR_sugar.fit(x_train, y_train)

print('\n\n')
print("GR Sugar Node Count:",GR_sugar.best_estimator_.tree_.node_count)
print("GR Sugar Max Depth:",GR_sugar.best_estimator_.tree_.max_depth)

y_train_pred_gr_sugar = GR_sugar.predict(x_train)
y_test_pred_gr_sugar = GR_sugar.predict(x_test)

train_test_full_error_gr_sugar = pd.Series({'Train': mean_squared_error(y_train,y_train_pred_gr_sugar),
                                            'Test': mean_squared_error(y_test,y_test_pred_gr_sugar)},name='MSE').to_frame().T


print('\n\n')
print(train_test_full_error_gr_sugar)

sns.set_context('notebook')
sns.set_style('white')
fig = plt.figure(figsize=(6,6))
ax = plt.axes()

ph_test_predict = pd.DataFrame({'test':y_test.values,
                                'predict': y_test_pred_gr_sugar}).set_index('test').sort_index()

ph_test_predict.plot(marker='o', ls='', ax=ax)
ax.set(xlabel='Test', ylabel='Predict', xlim=(0,35), ylim=(0,35))

plt.show()


'''
dot_data = StringIO()

export_graphviz(dt,out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

filename = 'Wine_Tree.png'
graph.write_png(filename)
Image(filename = filename)


dot_data = StringIO()

export_graphviz(GR.best_estimator_,out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

filename = 'Wine_Tree_Prune.png'
graph.write_png(filename)
Image(filename = filename)

'''