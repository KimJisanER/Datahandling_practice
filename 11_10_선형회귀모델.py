import warnings
warnings.filterwarnings("ignore")

import pandas as pd

data2=pd.read_csv('house_price.csv', encoding='utf-8')
X=data2[data2.columns[1:5]]
y=data2[["house_value"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_scaled_train=scaler.transform(X_train)
X_scaled_test=scaler.transform(X_test)

import statsmodels.api as sm
x_train_new= sm.add_constant(X_train)
x_test_new= sm.add_constant(X_test)
print(x_train_new.head())

multi_model= sm.OLS(y_train,x_train_new).fit()
print(multi_model.summary())

multi_model=sm.OLS(y_test,x_test_new).fit()
print(multi_model2.summary())