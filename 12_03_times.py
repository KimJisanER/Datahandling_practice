# 주어진 데이터에서 2022년 월별 Sales 합계 중 가장 큰 금액과
# 2023년 월별 Sales 합계 중 가장 큰 금액의 차이를 절대값으로 구하시오.
# 단, Events컬럼이 '1'인 경우 80%의 Salse값만 반영함
# (최종값은 소수점 반올림 후 정수 출력)
# 데이터셋 : basic2.csv

import pandas as pd
df=pd.read_csv('basic2.csv')

print(df)

for i in range(len(df)):
    if df.loc[i,'Events']==1:
        df.loc[i,'Sales2']=0.8*df.loc[i,'Sales']
    else :
        df.loc[i, 'Sales2'] = df.loc[i, 'Sales']

print(df)

df['Date']=pd.to_datetime(df['Date'])
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
# df['day']=df['Date'].dt.day
df_22=df[df['year']==2022]
df_22=df_22.groupby(['month'])['Sales2'].sum().max()
df_23=df[df['year']==2023]
df_23=df_23.groupby(['month'])['Sales2'].sum().max()

print(df_22)
print(df_23)

print(int(round(df_23-df_22,0)))