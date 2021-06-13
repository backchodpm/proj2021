import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

df1 = pd.DataFrame(pd.read_csv(r"MOE - in (1).csv"))
df1 = df1.dropna()
df1 = df1.drop(df1.columns.difference(['g', 'h', 'i', 'j', 'hu']), axis=1)
df1 = df1.dropna()
df1.hu[df1.hu == 'CONT'] = 0
df1.hu[df1.hu == 'EDP'] = 1
df1.hu[df1.hu == 'EL'] = 2
df1.hu[df1.hu == 'EL/EDP'] = 3
Y = df1['hu'].values
Y=Y.astype('int')
X = df1.drop(labels=['hu'], axis=1)
X=X.astype('float')
Y = np.array(Y)
X = np.array(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=20)
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
]
pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
