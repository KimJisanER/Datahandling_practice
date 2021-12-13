#주어진 데이터에서 결측치가 80%이상 되는 컬럼은(변수는) 삭제하고, 80% 미만인 결측치가 있는 컬럼은 'city'별 중앙값으로 값을 대체하고
#'f1'컬럼의 평균값을 출력하세요!

import pandas as pd
df=pd.read_csv('basic1.csv')

print(df)
print(df.shape)
print(df.isna().sum())

df=df.drop(columns=['f3'])
print(df)
# help(df.drop)
# df_gp=df.groupby()
# help(df.groupby)
a,b,c,d=df.groupby(by=['city'])['f1'].mean()

df['f1']=df['f1'].fillna(df['city'].map({'경기':a,'대구':b,'부산':c,'서울':d}))
print(df)

print(df['f1'].mean())