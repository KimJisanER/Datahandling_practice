
import pandas as pd

pd.set_option('display.max_columns',8)

X_train=pd.read_csv('bike_train.csv')
X_test=pd.read_csv('bike_test.csv')

X_train.loc[:,'datetime']=pd.to_datetime(X_train.loc[:,'datetime'])
X_test.loc[:,'datetime']=pd.to_datetime(X_test.loc[:,'datetime'])
print(X_train.T)

#dt 쪼개주기
X_train['year']=X_train['datetime'].dt.year
X_train['month']=X_train['datetime'].dt.month
X_train['day']=X_train['datetime'].dt.day
X_train['hour']=X_train['datetime'].dt.hour

X_test['year']=X_test['datetime'].dt.year
X_test['month']=X_test['datetime'].dt.month
X_test['day']=X_test['datetime'].dt.day
X_test['hour']=X_test['datetime'].dt.hour
# X_train[:,'day']=X_train[:,'datetime'].datetime.day
# print(X_train.head().T)

#dt 컬럼 저장
X_train_dt=X_train.loc[:,'datetime']
X_test_dt=X_test.loc[:,'datetime']
#y_train 추출
y_train=X_train.loc[:,'count']

#dt컬럼 삭제
X_train=X_train.drop(columns=['datetime','count','casual','registered'])
X_test=X_test.drop(columns=['datetime'])


#쪼개주기
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

#모델학습
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,y_train)
# model.fit(nX_train,ny_train)
# print(model.score(nX_train,ny_train))
# print(model.score(nX_test,ny_test))

# #예측하기
pred=model.predict(X_test)
# print(pred)
pred=pd.DataFrame(pred)
sub=pd.concat([X_test_dt,pred],axis=1)
sub.columns=['datetime','count']
# print(sub)
sub.to_csv('11_27_bike_regress.csv',index=False)


print(pd.read_csv('11_27_bike_regress.csv').head())
