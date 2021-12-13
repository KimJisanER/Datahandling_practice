# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name

    if null_name != "":
        df[df == null_name] = np.nan

    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test


df = pd.read_csv("bike_train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='count')  # , id_name='Id')

#사용자 코딩

# print(X_train.info())
# print(X_test.info())
pd.set_option('display.max_columns',12)

#id 빼놓기

test_id=X_test.index
test_id=pd.DataFrame(test_id)

y_train=y_train.drop(columns='id')

#datetime

X_train['datetime']=pd.to_datetime(X_train['datetime'])
X_test['datetime']=pd.to_datetime(X_test['datetime'])

X_train['year']=X_train['datetime'].dt.year
X_train['month']=X_train['datetime'].dt.month
X_train['day']=X_train['datetime'].dt.day
X_train['hour']=X_train['datetime'].dt.hour
X_train['dow']=X_train['datetime'].dt.dayofweek

X_test['year']=X_test['datetime'].dt.year
X_test['month']=X_test['datetime'].dt.month
X_test['day']=X_test['datetime'].dt.day
X_test['hour']=X_test['datetime'].dt.hour
X_test['dow']=X_test['datetime'].dt.dayofweek

X_train=X_train.drop(columns=['datetime','temp','workingday'])
X_test=X_test.drop(columns=['datetime','temp','workingday'])

#split
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.15,random_state=42)

#model
print('RF')
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,y_train.values.ravel())
# print(model.score(X_train,y_train))

#
print('XGB')
import xgboost as xgb
from xgboost import XGBRegressor
xmodel=XGBRegressor()
xmodel.fit(X_train,y_train.values.ravel())
print(xmodel.score(X_train,y_train))

pred=(xmodel.predict(X_test)+model.predict(X_test))/2
pred=pd.DataFrame(pred)

sub=pd.concat([test_id,pred],axis=1)
sub.columns=['id','count']
sub=sub.set_index('id',drop=False,inplace=False)
# print(sub)
sub.to_csv('12_02_bike.csv',index=False)

print(pd.read_csv('12_02_bike.csv').head(10))
print(y_test)
# help(sub.set_index)

from sklearn.metrics import mean_squared_error

print(round(np.sqrt(mean_squared_error(y_test['count'], pred))))


