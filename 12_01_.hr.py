# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가


# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("hr_data_X_test.csv")
X_train = pd.read_csv("hr_data_X_train.csv")
y_train = pd.read_csv("hr_data_y_train.csv")

# 사용자 코딩
pd.set_option('display.max_columns',15)

X_test_id=X_test.loc[:,'enrollee_id']

#컬럼 정리
y_train=y_train.drop(columns=['enrollee_id'])
X_train=X_train.drop(columns=['enrollee_id'])
X_test=X_test.drop(columns=['enrollee_id'])

print(X_train.describe())
#이상치 처리
# q1=X_train['training_hours'].quantile(q=0.25)
# q3=X_train['training_hours'].quantile(q=0.75)
# IQR=q3-q1

import numpy as np
Q1 = np.percentile(X_train['city_development_index'],25)
Q3 = np.percentile(X_train['city_development_index'],75)
IQR = Q3 - Q1
outdata1 = X_train[X_train['city_development_index']<(Q1 - 1.5 * IQR)]
outdata2 = X_train[X_train['city_development_index']>(Q3 + 1.5 * IQR)]

# print(X_train.shape)
ind = X_train[X_train['city_development_index']<(Q1 - 1.5 * IQR)].index
X_train = X_train.drop(index=ind, axis=0)
y_train = y_train.drop(index=ind, axis=0)

X_train = X_train.fillna("X")
X_test = X_test.fillna("X")

#라벨인코딩
obj_cols = np.array(X_train.columns[X_train.dtypes == object])
from sklearn.preprocessing import LabelEncoder

all_df = pd.concat([X_train.assign(ind="train"), X_test.assign(ind="test")])
le = LabelEncoder()
all_df[obj_cols] = all_df[obj_cols].apply(le.fit_transform)

X_train = all_df[all_df['ind'] == 'train']
X_train = X_train.drop('ind',axis=1)
X_train

X_test = all_df[all_df['ind'] == 'test']
X_test = X_test.drop('ind',axis=1)
X_test
# lab=['city','relevent_experience','enrolled_university','education_level','major_discipline','company_size','experience',
#      'company_type','last_new_job','gender']
#
# from sklearn.preprocessing import LabelEncoder
# X_train.loc[:,lab]=X_train.loc[:,lab].apply(LabelEncoder().fit_transform)
# X_test.loc[:,lab]=X_test.loc[:,lab].apply(LabelEncoder().fit_transform)

# print(X_train.head(3))

#split
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

#scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# scaler.fit(X_train)
# scaler.transform(X_train)
# scaler.transform(X_test)

# 스케일링
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

n_cols = ['city_development_index', 'training_hours']
X_train[n_cols] = scaler.fit_transform(X_train[n_cols])
X_test[n_cols] = scaler.transform(X_test[n_cols])
X_train

#model

# from sklearn.ensemble import RandomForestClassifier
# model=RandomForestClassifier(random_state=2022)
# model.fit(X_train,y_train.values.ravel())
# print(model.score(nX_train,ny_train))
# print(model.score(nX_test,ny_test))

# print('xgb')
import xgboost as xgb
from xgboost import XGBRFClassifier
xmodel=XGBRFClassifier(use_label_encoder=False,eval_metric='logloss')
xmodel.fit(X_train,y_train.values.ravel())
# print(xmodel.score(nX_train,ny_train.values.ravel()))
# print(xmodel.score(nX_test,ny_test))

pred=xmodel.predict(X_test)
pred=pd.DataFrame(pred)
sub=pd.concat([X_test_id,pred],axis=1)
sub.columns=['enrollee_id','target']
sub.to_csv('12_01_제출.csv', index=False)
#

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'enrollee_id': X_test_id, 'target': pred}).to_csv('12_01_제출.csv', index=False)

print(pd.read_csv('12_01_제출.csv').head())


# 체점
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

with open( "anwser.pickle", "rb" ) as file:
    ans = pickle.load(file)
    ans = pd.DataFrame(ans)
print(roc_auc_score(ans['target'], pred))