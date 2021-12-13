#f1의 결측치를 채운 후 age 컬럼의 중복 제거 전과 후의 중앙값의 차이 구하시오
#- 결측치는 f1의 데이터 중 10번째 큰 값으로 채움
#- 중복 데이터 발생시 뒤에 나오는 데이터를 삭제함
#- 최종 결과값은 절대값으로 출력

import pandas as pd
df=pd.read_csv('basic1.csv')

df_2=df.sort_values('f1',ascending=False).reset_index(drop=False)
# print(df_2.head(10))
top10=df_2.loc[9,'f1']
# print(top10)
df['f1'].fillna(top10,inplace=True)
a=df['f1'].median()
df=df.drop_duplicates(subset='age')
b=df['f1'].median()

print(abs(a-b))

