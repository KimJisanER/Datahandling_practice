# 정규성이나 등분산성 가정을 만족하지 않는 경우 처리하는 방법

#주어진 데이터에서 20세 이상인 데이터를 추출하고 'f1'컬럼을 결측치를 최빈값으로 채운 후,
#f1 컬럼의 여-존슨과 박스콕스 변환 값을 구하고, 두 값의 차이를 절대값으로 구한다음 모두 더해 소수점 둘째 자리까지 출력(반올림)하시오

from sklearn.preprocessing import power_transform
data = [[11, 12], [23, 22], [34, 35]]
print(power_transform(data)) # method 디폴트 값은 여-존슨’yeo-johnson’
print(power_transform(data, method='box-cox'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform

df = pd.read_csv('basic1.csv')
df.head(5)

print("조건 적용 전:", df.shape)
df = df[df['age']>=20]
print("조건 적용 후:", df.shape)

# 최빈값으로 'f1' 컬럼 결측치 대체
print("결측치 처리 전: \n", df.isnull().sum())
print("최빈값: ",df['f1'].mode()[0])
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])
print("결측치 처리 후: \n", df.isnull().sum())

# 'f1'데이터 여-존슨 yeo-johnson 값 구하기
df['y'] = power_transform(df[['f1']]) # method 디폴트 값은 여-존슨’yeo-johnson’
df['y'].head()

# 'f1'데이터 여-존슨 yeo-johnson 값 구하기
df['y'] = power_transform(df[['f1']],standardize=False) # method 디폴트 값은 여-존슨’yeo-johnson’
df['y'].head()

# 'f1'데이터 박스-콕스 box-cox 값 구하기
df['b'] = power_transform(df[['f1']], method='box-cox')
df['b'].head()

# 'f1'데이터 박스-콕스 box-cox 값 구하기
df['b'] = power_transform(df[['f1']], method='box-cox', standardize=False)
df['b'].head()

# 두 값의 차이를 절대값으로 구한다음 모두 더해 소수점 둘째 자리까지 출력(반올림)
print(round(sum(np.abs(df['y'] - df['b'])),2))