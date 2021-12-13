################## 시험 안내 문구 및 코드 ##################
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

test_id=X_test.loc[:,'enrollee_id']
drop_col=['enrollee_id','gender','company_size','company_type','major_discipline']
X_train=X_train.drop(columns=drop_col)
X_test=X_test.drop(columns=drop_col)
y_train=y_train.drop(columns='enrollee_id')

#라벨인코딩
from sklearn.preprocessing import LabelEncoder
lab=['city','relevent_experience','enrolled_university','education_level','experience','last_new_job']
X_train.loc[:,lab]=X_train.loc[:,lab].apply(LabelEncoder().fit_transform)
X_test.loc[:,lab]=X_test.loc[:,lab].apply(LabelEncoder().fit_transform)

# #split
# from sklearn.model_selection import train_test_split
# nX_train,nX_test,ny_train,ny_test=train_test_split(X_train,y_train,test_size=0.1,random_state=42)

#model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(max_depth=7,n_estimators=200)
model.fit(X_train,y_train.values.ravel())
# print(model.score(nX_train,ny_train))
# print(model.score(nX_test,ny_test))

#model2
import xgboost as xgb
from xgboost import XGBRFClassifier
xmodel=XGBRFClassifier(max_depth=8,use_label_encoder=False)
xmodel.fit(X_train,y_train.values.ravel())
# print(xmodel.score(nX_train,ny_train))
# print(xmodel.score(nX_test,ny_test))

pred1=model.predict_proba(X_test)
pred2=xmodel.predict_proba(X_test)

pred=(pred1+pred2)/2
pred=pd.DataFrame(pred)
pred=pred.iloc[:,1]

sub=pd.concat([test_id,pred],axis=1)
sub.columns=['enrollee_id','target']
sub.to_csv('12_02_mok2.csv',index=False)

print(pd.read_csv('12_02_mok2.csv'))

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'enrollee_id': X_test.enrollee_id, 'target': pred}).to_csv('003000000.csv', index=False)

import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

with open( "anwser.pickle", "rb" ) as file:
    ans = pickle.load(file)
    ans = pd.DataFrame(ans)
print(roc_auc_score(ans['target'], pred))