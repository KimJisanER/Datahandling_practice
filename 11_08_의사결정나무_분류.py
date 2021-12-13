import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data=pd.read_csv('breast-cancer-wisconsin.csv', encoding='utf-8')
X=data[data.columns[1:10]]
y=data[["Class"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_scaled_train=scaler.transform(X_train)
X_scaled_test=scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_scaled_train, y_train)
pred_train=model.predict(X_scaled_train)
print(model.score(X_scaled_train,y_train))

from sklearn.metrics import confusion_matrix
confusion_train=confusion_matrix(y_train,pred_train)
print("훈련데이터 오차행렬:\n", confusion_train)

from sklearn.metrics import classification_report
cfreport_train=classification_report(y_train,pred_train)
print("분류예측 레포트:\n", cfreport_train)

pred_test=model.predict(X_scaled_test)
print(model.score(X_scaled_test, y_test))

confusion_test=confusion_matrix(y_test, pred_test)
print("테스트데이터 오차행렬:\n", confusion_test)

from sklearn.metrics import classification_report
cfreport_test=classification_report(y_test,pred_test)
print("분류예측 레포트:\n", cfreport_test)

param_grid={'max_depth': range(2,20,2), 'min_samples_leaf': range(1,50,2)}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(DecisionTreeClassifier(),param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(grid_search.best_params_))
print("Best Score: {:.4f}".format(grid_search.best_score_))
print("TestSet Score: {:.4f}".format(grid_search.score(X_scaled_test, y_test)))

from scipy.stats import randint
param_distribs = {'max_depth': randint(low=1, high=20), 'min_samples_leaf': randint(low=1, high=50)}
from sklearn.model_selection import RandomizedSearchCV
random_search=RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=param_distribs, n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter: {}".format(random_search.best_params_))
print("Best Score: {:.4f}".format(random_search.best_score_))
print("TestSet Score: {:.4f}".format(random_search.score(X_scaled_test,y_test)))
