import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Iris 데이터셋 불러오기
from sklearn import datasets
datasets = datasets.load_iris()

# 데이터 전처리
# 데이터셋의 모든 이미지를 tensor로 변환
# 데이터셋의 모든 라벨 또한 tensor로 변환환
import torch
X = torch.tensor(datasets.data, dtype=torch.float32)
y = torch.tensor(datasets.target)

# MLP 모델 선언
# 은닉층의 뉴런수(hidden_units)를 입력으로 받음
# 1개의 은닉층을 가짐
# ReLU 활성화 함수를 사용
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_units):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# MLP 모델 인스턴스화
# 은닉층이 5개의 새로운 특징을 만들어내는 MLP 모델을 인스턴스화화
model = MLP(1000)


# 경사하강법과 손실함수 설정
# LR이 0.1인 SGD(Stochastic Gradient Descent) 알고리즘을 사용
# 모델의 파라미터들을 대상으로 경사하강을 진행함을 설정
# Cross-Entropy 손실함수를 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# 모델 학습
# 100번을 반복해서 학습
# 모델에 샘플들(X)을 입력, 정답을 예측(y_pred)
# 실제 정답(y)과 비교, 손실을 계산
# 손실을 기반으로 역전파
# 경사하강법으로 파라미터 업데이트
def train(model, optimizer, criterion):
    for epoch in range(100):
        y_pred = model(X)

        loss = criterion(y_pred, y)
        print(f'Epoch {epoch}, Loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 함수 외부에서 train 함수 호출
train(model, optimizer, criterion)