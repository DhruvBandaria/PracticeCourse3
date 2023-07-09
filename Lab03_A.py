import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

data = pd.read_csv('./data/Human_Activity_Recognition_Using_Smartphones_Data.csv')

print(data.dtypes.value_counts())

print('\n')
print(data.Activity.value_counts())

le = LabelEncoder()

data['Activity'] = le.fit_transform(data.Activity)

feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

tril_index = np.tril_indices_from(corr_values)

corr_array = np.array(corr_values)
corr_array[np.tril_indices_from(corr_values)] = np.nan

corr_values = pd.DataFrame(corr_array,columns= corr_values.columns, index= corr_values.index)

corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0':'feature1',
                                'level_2':'feature2',
                                0:'correlation'}))

corr_values['abs_correlation'] = corr_values.correlation.abs()

strat_shuf_split = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols],data.Activity))

x_train = data.loc[train_idx,feature_cols]
y_train = data.loc[train_idx,'Activity']

x_test = data.loc[test_idx,feature_cols]
y_test = data.loc[test_idx,'Activity']

lr = LogisticRegression(solver='liblinear').fit(x_train,y_train)

lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(x_train,y_train)

lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(x_train,y_train)


coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 
                                 codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)

print('\n\n')
print(coefficients.sample(10))


y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(x_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(x_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

print('\n\n')
print(y_pred.head())

print('\n\n')
print(y_prob.head())

metrics = list()
cm = dict()

for lab in coeff_labels:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

print('\n\n')
print(metrics)



