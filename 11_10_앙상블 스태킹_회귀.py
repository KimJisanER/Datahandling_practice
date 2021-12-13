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

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
estimators=[('ir', LinearRegression()),('knn', KNeighborsRegressor())]
model=StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
model.fit(X_scaled_train, y_train)
pred_train=model.predict(X_scaled_train)
print(model.score(X_scaled_train, y_train))

pred_test=model.predict(X_scaled_test)
print(model.score(X_scaled_test, y_test))
