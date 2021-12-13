# 이상치를 찾아라(소수점 나이)
# 주어진 데이터에서 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 3가지 모두 이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오¶
#
# 데이터셋 : basic1.csv
# 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작

import pandas as pd
import numpy as np
df=pd.read_csv('basic1.csv')

out=df[df['age']-round(df['age'],0)!=0]

a=np.ceil(out['age']).mean()
b=np.floor(out['age']).mean()
c=np.trunc(out['age']).mean()

print(a)
print(b)
print(c)
print(a+b+c)