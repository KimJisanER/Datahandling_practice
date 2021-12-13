# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
# 앞에서 부터 20개의 데이터를 추출한 후
# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)
# 데이터셋 : basic1.csv
#
# 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
#
# File -> Editor Type -> Script
#
# 정답 : 38.44

import pandas as pd
df=pd.read_csv('basic1.csv')
df=df[df['f2']==0]
df=df.sort_values('age',ascending=True)
df=df.head(20)
a=df['f1'].var()
print(df)
df['f1'].fillna(df['f1'].min(),inplace=True)
b=df['f1'].var()
print(df)
print(b)
print(a)
print(round(a-b,2))