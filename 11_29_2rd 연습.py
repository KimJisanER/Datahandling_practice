import pandas as pd
import numpy as np

X_train=pd.read_csv('2rd_X_train.csv')
y_train=pd.read_csv('2rd_y_train.csv')
X_test=pd.read_csv('2rd_X_test.csv')

pd.set_option('display.max_columns',12)


#test id 빼놓기
test_id=X_test.index
test_id=pd.DataFrame(test_id)
y_train=y_train.drop(columns=['ID'])

#encoding
#onehot
X_train.loc[:,'Gender']=X_train.loc[:,'Gender'].replace('M',0).replace('F',1)
X_test.loc[:,'Gender']=X_test.loc[:,'Gender'].replace('M',0).replace('F',1)
#label
from sklearn.preprocessing import LabelEncoder
X_train.loc[:,['Warehouse_block','Mode_of_Shipment','Product_importance']]\
    =X_train.loc[:,['Warehouse_block','Mode_of_Shipment','Product_importance']].apply(LabelEncoder().fit_transform)


from sklearn.model_selection import train_test_split
nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(nX_train)
scaler.transform(nX_train)
scaler.transform(nX_test)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=7,n_estimators=100)
model.fit(nX_train,ny_train.values.ravel())
#
print(model.score(nX_train,ny_train))
print(model.score(nX_test,ny_test))

# print(X_train.head().T)
# print(X_test.head().T)
# print(y_train.head().T)