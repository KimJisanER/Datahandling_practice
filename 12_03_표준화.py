#수치형 변수 변환하기
# 주어진 데이터에서 'f5'컬럼을 표준화(Standardization (Z-score Normalization))하고 그 중앙값을 구하시오
# 데이터셋 : basic1.csv
# 노트북 선택 - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작

import pandas as pd
import numpy as np
df=pd.read_csv('basic1.csv')

print(df['f5'])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df['f5']=scaler.fit_transform(df[['f5']])
# scaler.fit(df['f5'])
# scaler.transform(df['f5'])
# scaler.fit(df['f5'])

print(df['f5'])