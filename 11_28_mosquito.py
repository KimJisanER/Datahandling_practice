
import pandas as pd
X_train =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_x.csv',encoding='euc-kr')
y_train =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/train_y.csv',encoding='euc-kr')
X_test =pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/test_x.csv',encoding='euc-kr')

Ans=pd.read_csv('https://raw.githubusercontent.com/Datamanim/mosquito/main/sub.csv')

#date 쪼개기
X_train['date']=pd.to_datetime(X_train['date'])
X_test['date']=pd.to_datetime(X_test['date'])

X_train['year']=X_train['date'].dt.year
X_train['month']=X_train['date'].dt.month
X_train['day']=X_train['date'].dt.day

X_test['year']=X_test['date'].dt.year
X_test['month']=X_test['date'].dt.month
X_test['day']=X_test['date'].dt.day

#date_test 쪼개기
X_test_dt=X_test.loc[:,'date']

#drop columns
X_train=X_train.drop(columns=['date'])
X_test=X_test.drop(columns=['date'])
y_train=y_train.drop(columns=['date'])

#data 셋 나누기
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train, test_size=0.2, random_state=42)

#model 학습

#1
from sklearn.ensemble import RandomForestRegressor
Rfmodel=RandomForestRegressor()
Rfmodel.fit(X_train,y_train.values.ravel())
pred_rf=Rfmodel.predict(X_test)
# print(Rfmodel.score(nX_train,ny_train))
# print(Rfmodel.score(nX_test,ny_test))

#2
import xgboost as xgb
from xgboost import XGBRFRegressor
Xgmodel=XGBRFRegressor()
Xgmodel.fit(X_train,y_train.values.ravel())
pred_xg=Xgmodel.predict(X_test)
# print(Xgmodel.score(nX_train,ny_train))
# print(Xgmodel.score(nX_test,ny_test))

#모델합치기
pred=(pred_rf+pred_xg)/2

#평가
from sklearn.metrics import r2_score

# print('rf:',r2_score(ny_test,pred_rf))
# print('xg:',r2_score(ny_test,pred_xg))
# print('mix:',r2_score(ny_test,pred))

#제출
pred=pd.DataFrame(pred)
submis=pd.concat([X_test_dt,pred], axis=1)
submis.columns=['date','mosqutio_ratio']
print(submis)
submis.to_csv('11_28_mosq.csv',index=False)

print(pd.read_csv('11_28_mosq.csv'))

#시간재기
import math
import time
start =time.time()
math.factorial(100000)
end = time.time()
print(f"{end - start:.5f} sec")


