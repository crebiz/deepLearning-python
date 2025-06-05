import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델 불러오기
model = load_model('data/model/house_train_boston_model.keras')

# 모델 정보 출력
model.summary()

# 가중치 정보 출력
for layer in model.layers:
    weights = layer.get_weights()
    print(f"레이어: {layer.name}, 가중치 형태: {[w.shape for w in weights]}")

# 테스트 데이터 준비 (예시)
# 실제 사용 시 테스트 데이터를 적절히 준비해야 함
df = pd.read_csv("data/house_train.csv")
df = pd.get_dummies(df)
df = df.fillna(df.mean())

# 필요한 특성 선택
cols_train = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
X_test = df[cols_train].values[:20]  # 예시로 20개 샘플만 사용
y_test = df['SalePrice'].values[:20]

# 예측 수행
y_pred = model.predict(X_test).flatten()

# 결과 출력
for i in range(len(y_test)):
    print(f"실제값: {y_test[i]:.2f}, 예측값: {y_pred[i]:.2f}, 오차: {y_test[i] - y_pred[i]:.2f}")

# MSE 계산
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('실제 가격')
plt.ylabel('예측 가격')
plt.title('실제 가격 vs 예측 가격')
plt.show()

# 오차 분포 시각화
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
plt.hist(errors, bins=20)
plt.xlabel('예측 오차')
plt.ylabel('빈도')
plt.title('예측 오차 분포')
plt.show()