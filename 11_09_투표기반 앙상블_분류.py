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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

logit_model=LogisticRegression(random_state=42)
rnf_model=RandomForestClassifier(random_state=42)
svm_model=SVC(random_state=42)

voting_hard= VotingClassifier(estimators=[('ir',logit_model), ('rf', rnf_model), ('svc', svm_model)], voting='hard')
voting_hard.fit(X_scaled_train,y_train)

from sklearn.metrics import accuracy_score

for clf in (logit_model, rnf_model, svm_model ,voting_hard):
    clf.fit(X_scaled_train,y_train)
    y_pred = clf.predict(X_scaled_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

logit_model=LogisticRegression(random_state=42)
rnf_model=RandomForestClassifier(random_state=42)
svm_model=SVC(random_state=42)
voting_soft= VotingClassifier(estimators=[('ir',logit_model), ('rf', rnf_model), ('svc', svm_model)], voting='soft')
voting_soft.fit(X_scaled_train,y_train)

from sklearn.metrics import accuracy_score
for clf in (logit_model, rnf_model, svm_model ,voting_soft):
    clf.fit(X_scaled_train,y_train)
    y_pred = clf.predict(X_scaled_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

# from sklearn.metrics import confusion_matrix
# log_pred_train=logit_model.predict(X_scaled_train)
# log_confusion_train=confusion_matrix(y_train, log_pred_train)
# print("로지스틱 분류기 훈련데이터 오차행렬:\n", log)