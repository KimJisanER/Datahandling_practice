import pandas as pd

X_train=pd.read_csv('income_X_train.csv')
y_train=pd.read_csv('income_y_train.csv')
X_test=pd.read_csv('income_X_test.csv')
y_test=pd.read_csv('income_y_test.csv')

pd.set_option('display.max_columns',15)

print(y_train)
y_train=y_train.drop(columns=['id'])

#인코딩
X_train.loc[:,'sex']=X_train.loc[:,'sex'].replace('Male',0).replace('Female',1)
X_test.loc[:,'sex']=X_test.loc[:,'sex'].replace('Male',0).replace('Female',1)

from sklearn.preprocessing import LabelEncoder
z=['workclass','marital.status','occupation','relationship','race','native.country']
X_train.loc[:,z]=X_train.loc[:,z].apply(LabelEncoder().fit_transform)
X_test.loc[:,z]=X_test.loc[:,z].apply(LabelEncoder().fit_transform)

y_train.loc[:,['income']]=y_train.loc[:,['income']].apply(LabelEncoder().fit_transform)

#drop
X_train=X_train.drop(columns=['education'])
X_test=X_test.drop(columns=['education'])

#Train_test

# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

#스케일링
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

#모델링
#1
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=7,n_estimators=100)
model.fit(X_train,y_train.values.ravel())
print('RF')


#2
import xgboost as xgb
# from xgboost import XGBRFClassifier
# xmodel=XGBRFClassifier(use_label_encoder=False)
# xmodel.fit(X_train,y_train.values.ravel())
# print('xgb')
# print(xmodel.score(nX_train,ny_train))
# print(xmodel.score(nX_test,ny_test))

pred=model.predict(X_test)
pred=pd.DataFrame(pred)
test_id=y_test['id']
ans=y_test.drop(columns=['id'])

from sklearn.metrics import roc_auc_score
print(roc_auc_score(ans,pred))
pred.columns=['income']
pred=pred.loc[:,'income'].replace(0,'<=50K').replace(1,'>50K')

sub=pd.concat([test_id,pred],axis=1)
# sub=sub.reset_index().rename(columns={"index": "id"})
sub=sub.set_index('id',drop=False)
print(sub)

# help(sub.reset_index)