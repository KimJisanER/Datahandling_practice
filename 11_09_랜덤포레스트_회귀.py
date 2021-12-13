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

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_scaled_train, y_train)
pred_train=model.predict(X_scaled_train)
print(model.score(X_scaled_train, y_train))

pred_test=model.predict(X_scaled_test)
print(model.score(X_scaled_test, y_test))

#RMSE
import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE:", np.sqrt(MSE_train))
print("테스트 데이터 RMSE:", np.sqrt(MSE_test))

#Random Search
from scipy.stats import randint
param_distribs={'n_estimators': randint(low=100, high=500), 'max_features':['auto','sqrt','log2']}
from sklearn.model_selection import RandomizedSearchCV
random_search=RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_distribs, n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("TestSet Score: {:.4f}".format(random_search.score(X_scaled_test,y_test)))

import time
start=time.time()
print(time.time()-start)