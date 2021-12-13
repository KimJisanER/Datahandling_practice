import time
start =time.time()

import pandas as pd

X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/jeju/main/Jeju_trainX.csv",encoding='euc-kr')
y_train= pd.read_csv("https://raw.githubusercontent.com/Datamanim/jeju/main/Jeju_trainy.csv",encoding='euc-kr')
X_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/jeju/main/Jeju_testX.csv",encoding='euc-kr')
sub= pd.read_csv("https://raw.githubusercontent.com/Datamanim/jeju/main/subExample.csv",encoding='euc-kr')

pd.set_option('display.max_columns',15)

X_test_id=X_test.loc[:,'id']

#date
X_train.loc[:,'일자']=pd.to_datetime(X_train.loc[:,'일자'])
X_train.loc[:,'year']=X_train.loc[:,'일자'].dt.year
X_train.loc[:,'month']=X_train.loc[:,'일자'].dt.month
X_train.loc[:,'day']=X_train.loc[:,'일자'].dt.day

X_test.loc[:,'일자']=pd.to_datetime(X_test.loc[:,'일자'])
X_test.loc[:,'year']=X_test.loc[:,'일자'].dt.year
X_test.loc[:,'month']=X_test.loc[:,'일자'].dt.month
X_test.loc[:,'day']=X_test.loc[:,'일자'].dt.day

#drop
X_train=X_train.drop(columns=['id','일자'])
#평균 풍속할지 말지
X_test=X_test.drop(columns=['id','일자'])
y_train=y_train.drop(columns=['id'])

#원핫, drop 할지말지 생각좀
X_train.loc[:,'시도명']=X_train.loc[:,'시도명'].replace('제주시',0).replace('서귀포시',1)
X_test.loc[:,'시도명']=X_test.loc[:,'시도명'].replace('제주시',0).replace('서귀포시',1)

#라벨인코딩
from sklearn.preprocessing import LabelEncoder
X_train.loc[:,['읍면동명']]=X_train.loc[:,['읍면동명']].apply(LabelEncoder().fit_transform)
X_test.loc[:,['읍면동명']]=X_test.loc[:,['읍면동명']].apply(LabelEncoder().fit_transform)

#스케일링
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

#데이터셋 나누기
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

#모델
#Rf
from sklearn.ensemble import RandomForestRegressor
Rfmodel=RandomForestRegressor()
Rfmodel.fit(X_train,y_train.values.ravel())

#xgb
import xgboost as xgb
from xgboost import XGBRFRegressor
Xgmodel=XGBRFRegressor()
Xgmodel.fit(X_train,y_train.values.ravel())

#
pred_rf=Rfmodel.predict(X_test)
pred_xg=Xgmodel.predict(X_test)
pred=(pred_rf+pred_xg)/2

#모델평가
# from sklearn.metrics import r2_score
# print(r2_score(ny_test,pred_rf))
# print(r2_score(ny_test,pred_xg))
# print(r2_score(ny_test,pred))

#제출물
pred=pd.DataFrame(pred)
sub=pd.concat([X_test_id,pred],axis=1)
sub.columns=['id','교통량']

#제출
sub.to_csv('11_28_jeju.csv',index=False)

# print(pd.read_csv('11_28_jeju.csv').head())


end = time.time()
print(f"{end - start:.5f} sec")
