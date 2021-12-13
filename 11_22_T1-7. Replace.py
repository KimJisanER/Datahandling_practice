#'f4'컬럼의 값이 'ESFJ'인 데이터를 'ISFJ'로 대체하고, 'city'가 '경기'이면서 'f4'가 'ISFJ'인 데이터 중 'age'컬럼의 최대값을 출력하시오!

import pandas as pd
df=pd.read_csv('basic1.csv')

print(df)

df[df['f4']=='ESFJ']=df[df['f4']=='ESFJ'].replace('ESFJ','ISFJ')
print(df)

print(df[(df['city']=='경기')&(df['f4']=='ISFJ')]['age'].max())