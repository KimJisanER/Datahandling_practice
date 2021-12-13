#city와 f4를 기준으로 f5의 평균값을 구한 다음, f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
# 데이터셋 : basic1.csv
# 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# File -> Editor Type -> Script

import pandas as pd
df=pd.read_csv('basic1.csv')

df=df.groupby(['city','f4'])['f5'].mean()
print(df)
print(df.shape)
df=df.reset_index()
df=df.sort_values('f5',ascending=False)
print(df)
print(df.shape)
df=df.head(7)
print(round(df['f5'].sum(),2))
