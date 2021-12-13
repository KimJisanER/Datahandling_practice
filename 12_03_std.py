# 주어진 데이터 중 basic1.csv에서 'f4'컬럼 값이 'ENFJ'와 'INFP'인 'f1'의 표준편차 차이를 절대값으로 구하시오
#
# 데이터셋 : basic1.csv

import pandas as pd
df=pd.read_csv('basic1.csv')

a=df[df['f4']=='ENFJ']['f1'].std()
b=df[df['f4']=='INFP']['f1'].std()

print(abs(b-a))