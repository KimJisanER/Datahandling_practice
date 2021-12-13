#주어진 데이터에서 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 3가지 모두 이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오
import pandas as pd
import numpy as np
df=pd.read_csv('basic1.csv')

print(df)
print(df.info())

#소수점나이 찾기

df=df[(df['age']-np.floor(df['age']))!=0]
print(df)

ceil=np.ceil(df['age']).mean()
floor=np.floor(df['age']).mean()
trunc=np.trunc(df['age']).mean()

print(ceil)
print(floor)
print(trunc)

print(ceil+floor+trunc)