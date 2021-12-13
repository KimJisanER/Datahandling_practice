# 주어진 데이터에서 2022년 5월 주말과 평일의 sales컬럼 평균값 차이를 구하시오 (소수점 둘째자리까지 출력, 반올림)
# 데이터셋 : basic2.csv
# 필사: 원하는 노트북 선택 - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작

import pandas as pd
df=pd.read_csv('basic2.csv')

print(df.info())
df['Date']=pd.to_datetime(df['Date'])
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
df['dayofweek']=df['Date'].dt.dayofweek

df=df[(df['year']==2022)&(df['month']==5)]
df_e=df[df['dayofweek']>=5]['Sales'].mean()
df_w=df[df['dayofweek']<5]['Sales'].mean()
print(df_e)
print(df_w)
print(abs(round(df_w-df_e,2)))


# print(round(30.175,2))