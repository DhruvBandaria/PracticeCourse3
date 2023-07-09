def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC,SVC

data = pd.read_csv('./data/Wine_Quality_Data.csv')
y = (data['color']=='red').astype(int)
fields = list(data.columns[:-1])
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace=True)
print(correlations)

'''
sns.set_context('talk')
sns.set_style('white')
sns.pairplot(data,hue='color')
plt.show()
'''

fields = correlations.map(abs).sort_values().iloc[-2:].index
print('\n\n')
print(fields)

x = data[fields]
scaler =MinMaxScaler()

x = scaler.fit_transform(x)
x = pd.DataFrame(x,columns=['%s_scaled' % fld for fld in fields])

print('\n\n')
print(x.columns)

sv = SVC(C=10,kernel='linear')

