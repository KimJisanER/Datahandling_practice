################## 시험 안내 문구 및 코드 ##################
# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("hr_data_X_test.csv")
X_train = pd.read_csv("hr_data_X_train.csv")
y_train = pd.read_csv("hr_data_y_train.csv")

####
print(X_train.corr())