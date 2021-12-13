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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
model=BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=10, random_state=0)
model.fit(X_scaled_train,y_train)
pred_train=model.predict(X_scaled_train)
print(model.score(X_scaled_train,y_train))

pred_test=model.predict(X_scaled_test)
print(model.score(X_scaled_test,y_test))

import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE:", np.sqrt(MSE_train))
print("테스트 데이터 RMSE:", np.sqrt(MSE_test))