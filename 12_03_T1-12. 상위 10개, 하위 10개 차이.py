# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요
# (단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력)

import pandas as pd

df = pd.read_csv("covid_vaccination_vs_death_ratio.csv")

print(df.head())

df=df[df['ratio']<100]
df=df.groupby(['country'])['ratio'].mean()

df=df.reset_index()
df=df.sort_values('ratio',ascending=False)

df_h=df.head(10)['ratio'].mean()
df_t=df.tail(10)['ratio'].mean()

print(round(df_h-df_t,3))

