import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv('./data/churndata_processed.csv')

print(data.describe().T)
#print(round(data.rename(columns=labels).describe().T,2))

target = 'churn_value'
print('\n')
print(data[target].value_counts())

print('\n')
print(data[target].value_counts(normalize=True))

feature_cols = [x for x in data.columns if x != target]

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=42)

train_idx, test_idx = next(strat_shuff_split.split(data[feature_cols],data[target]))

x_train = data.loc[train_idx,feature_cols]
y_train = data.loc[train_idx,target]

x_test = data.loc[test_idx,feature_cols]
y_test = data.loc[test_idx,target]

print('\n\n')
print(y_train.value_counts(normalize=True))
print('\n')
print(y_test.value_counts(normalize=True))


RF = RandomForestClassifier(oob_score=True, random_state=42, warm_start=True, n_jobs=-1)

oob_list = list()

for n_trees in [15,20,30,40,50,100,150,200,300,400]:
    RF.set_params(n_estimators = n_trees)
    RF.fit(x_train,y_train)

    oob_error = 1 - RF.oob_score_

    oob_list.append(pd.Series({'n_trees':n_trees, 'oob': oob_error}))

rf_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

print('\n\n')
print(rf_oob_df)

EF = ExtraTreesClassifier(oob_score=True,
                          random_state=10,
                          warm_start=True,
                          bootstrap=True,
                          n_jobs=1)


oob_list = list()

for n_trees in [15,20,30,40,50,100,150,200,300,400]:
    EF.set_params(n_estimators = n_trees)
    EF.fit(x_train,y_train)

    oob_error = 1 - EF.oob_score_

    oob_list.append(pd.Series({'n_trees':n_trees, 'oob': oob_error}))

ef_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')

print('\n\n')
print(ef_oob_df)


model = RF.set_params(n_estimators = 100)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)

print('\n\n')
print(y_pred)

cr = classification_report(y_test,y_pred)

score_df = pd.DataFrame({'accuracy': accuracy_score(y_test,y_pred),
                         'precision': precision_score(y_test,y_pred),
                         'recall': recall_score(y_test,y_pred),
                         'f1': f1_score(y_test,y_pred),
                         'auc': roc_auc_score(y_test,y_prob[:,1])}, index = pd.Index([0]))


print('\n\n')
print(score_df)

