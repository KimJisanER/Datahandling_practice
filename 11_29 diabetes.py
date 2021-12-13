import pandas as pd
df = pd.read_csv("diabetes.csv")

pd.set_option('display.max_columns',15)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
y_train = X_train['Outcome']
X_train = X_train.drop(columns=['Outcome'])
y_test = X_test['Outcome']
X_test = X_test.drop(columns=['Outcome'])

# print(X_train.describe())
# print(X_test.describe())

X_train.loc[:,'Glucose']=X_train.loc[:,'Glucose'].replace(0,X_train.loc[:,'Glucose'].median())
X_test.loc[:,'Glucose']=X_test.loc[:,'Glucose'].replace(0,X_test.loc[:,'Glucose'].median())

X_train.loc[:,'BloodPressure']=X_train.loc[:,'BloodPressure'].replace(0,X_train.loc[:,'BloodPressure'].median())
X_test.loc[:,'BloodPressure']=X_test.loc[:,'BloodPressure'].replace(0,X_test.loc[:,'BloodPressure'].median())

X_train.loc[:,'SkinThickness']=X_train.loc[:,'SkinThickness'].replace(0,X_train.loc[:,'SkinThickness'].median())
X_test.loc[:,'SkinThickness']=X_test.loc[:,'SkinThickness'].replace(0,X_test.loc[:,'SkinThickness'].median())

# X_train.loc[:,'Insulin']=X_train.loc[:,'Insulin'].replace(0,X_train.loc[:,'Insulin'].mean())
# X_test.loc[:,'Insulin']=X_test.loc[:,'Insulin'].replace(0,X_test.loc[:,'Insulin'].mean())

X_train.loc[:,'BMI']=X_train.loc[:,'BMI'].replace(0,X_train.loc[:,'BMI'].median())
X_test.loc[:,'BMI']=X_test.loc[:,'BMI'].replace(0,X_test.loc[:,'BMI'].median())

# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=4,n_estimators=200)
model.fit(X_train,y_train)

test_id=X_test.index
pred=model.predict(X_test)
test_id=pd.DataFrame(test_id)
pred=pd.DataFrame(pred)

sub=pd.concat([test_id,pred],axis=1)
sub.columns=['id','Outcome']

sub.to_csv('11_29_dib.csv',index=False)

print(pd.read_csv('11_29_dib.csv'))

Ans=y_test
print(Ans)
print(pred)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(Ans,pred))
# # print(sub)

# print(pred)

# import xgboost as xgb
# from xgboost import XGBClassifier
# xmodel=XGBClassifier(use_label_encoder=False)
# xmodel.fit(nX_train,ny_train)
#
# print(model.score(nX_train,ny_train))
# print(model.score(nX_test,ny_test))
#
# print(xmodel.score(nX_train,ny_train))
# print(xmodel.score(nX_test,ny_test))




