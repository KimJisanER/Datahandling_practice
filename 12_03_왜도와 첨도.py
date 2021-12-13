#주어진 데이터 중 train.csv에서 'SalePrice'컬럼의 왜도와 첨도를 구한 값과,
# 'SalePrice'컬럼을 스케일링(log1p)로 변환한 이후 왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오

import pandas as pd
import numpy as np
df = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")

print(df.head())
a=df['SalePrice'].skew()
b=df['SalePrice'].kurt()
print(df['SalePrice'].skew())
print(df['SalePrice'].kurt())

df['SalePrice']=np.log1p(df['SalePrice'])
print(df['SalePrice'].skew())
print(df['SalePrice'].kurt())

print(round(a+b+df['SalePrice'].skew()+df['SalePrice'].kurt(),2))