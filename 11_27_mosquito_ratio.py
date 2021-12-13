

import pandas as pd
X_train =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_x.csv',encoding='euc-kr')
y_train =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_y.csv',encoding='euc-kr')
X_test =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/test_x.csv',encoding='euc-kr')

Ans=pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/sub.csv')
pd.set_option('display.max_columns',10)


#datetime쪼개기
X_train.loc[:,'date']=pd.to_datetime(X_train.loc[:,'date'])
X_test.loc[:,'date']=pd.to_datetime(X_test.loc[:,'date'])

X_train.loc[:,'year']=X_train.loc[:,'date'].dt.year
X_train.loc[:,'month']=X_train.loc[:,'date'].dt.month
X_train.loc[:,'day']=X_train.loc[:,'date'].dt.day

X_test.loc[:,'year']=X_test.loc[:,'date'].dt.year
X_test.loc[:,'month']=X_test.loc[:,'date'].dt.month
X_test.loc[:,'day']=X_test.loc[:,'date'].dt.day


X_test_dt=X_test.loc[:,'date']

X_train=X_train.drop(columns=['date'])
X_test=X_test.drop(columns=['date'])
y_train=y_train.drop(columns=['date'])


#스케일링 안해도 상관없나?
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#데이터셋 분리

from sklearn.model_selection import train_test_split
nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2, random_state=42)

#모델적용
#1
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,y_train.values.ravel())
#2
import xgboost as xgb
# from xgboost import XGBRFRegressor
# model2=XGBRFRegressor()
# model2.fit(X_train,y_train.value.ravel())
#모델평가
from sklearn.metrics import r2_score

pred=model.predict(X_test)
pred=pd.DataFrame(pred)
# print(r2_score(ny_test,pred))

sub=pd.concat([X_test_dt,pred],axis=1)
sub.columns=['date','mosquito ratio']

# print(sub)
sub.to_csv('11_17_mosq.csv',index=False)

print(pd.read_csv('11_17_mosq.csv').head())
# help(RandomForestRegressor)

