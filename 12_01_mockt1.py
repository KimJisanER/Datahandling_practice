#1. 1. 첫번째 데이터 부터 순서대로 50:50으로 데이터를 나누고,
# 앞에서 부터 50%의 데이터(이하, A그룹)는 'f1'컬럼을 A그룹의 중앙값으로 채우고,
# 뒤에서부터 50% 데이터(이하, B그룹)는 'f1'컬럼을 B그룹의 최대값으로 채운 후, A그룹과 B그룹의 표준편차 합을 구하시오

import pandas as pd
import numpy as np
df=pd.read_csv('basic1.csv')

# df_A=df[:int(0.5*len(df))]
# df_B=df[int(0.5*len(df)):]
#
#
# df_A.fillna(df_A.loc[:,'f1'].median(),inplace=True)
# df_B.fillna(df_B.loc[:,'f1'].max(),inplace=True)
#
# ans=df_A.loc[:,'f1'].std()+df_B.loc[:,'f1'].std()
# print(round(ans,1))

# df=df.sort_values(['f4','f5'],ascending=['False','True']).reset_index(drop=True)
# # print(df)
# # print(min_f5)
# df.loc[:9,'f5']=df['f5'].head(10).min()
# # print(df.loc[:9,'f5'])
# print(round(df['f5'].mean(),2))

#3. 'age' 컬럼의 IQR방식을 이용한 이상치 수와 표준편차*1.5방식을 이용한 이상치 수 합을 구하시오

q1=df.loc[:,'age'].quantile(q=0.25)
q3=df.loc[:,'age'].quantile(q=0.75)
IQR=q3-q1
print(q3)
print(q1)
print(IQR)

IQ_out=df[(df.loc[:,'age']>=q3+1.5*IQR)|(df.loc[:,'age']<=q1-1.5*IQR)].loc[:,'age']
print(len(IQ_out))

ag_st=df.loc[:,'age'].std()
ag_me=df.loc[:,'age'].mean()
print(ag_st)
st_out=df[(df.loc[:,'age']>=ag_me+1.5*ag_st)|(df.loc[:,'age']<=ag_me-1.5*ag_st)]
print(len(st_out))

print(len(st_out)+len(IQ_out))