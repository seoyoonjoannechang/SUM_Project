# 익일 서울시 종로구 미세먼지 농도 예측

### 목표

실시간 미세먼지 예측 모델 개발
1. 서울, 인천, 경기의 기상관측, 대기오염 요소 중, 24시간 뒤 미세먼지 농도 예측에 가장 큰 영향력을 가진 요소 추출
2. 전날 기상, 대기오염 데이터로 24시간 뒤 서울시 종로구의 미세먼지 농도를 예측하는 모델 생성  
3. 실시간 24시간 뒤 미세먼지 농도 예측 범위 제공 웹사이트 생성

### 사용기술 및 개발환경
- 데이터 출처 : 기상청 API Hub, 공공데이터 포털 
- 데이터 수집 방법 : OPEN API 접속
- 개발도구 : matplotlib, seaborn, pyplot, scikit-learn, pandas, numpy, requests, json, time, tensorflow, keras
- 개발언어 및 프레임워크 : Python, HTML, CSS, Javascript, Bootstrap, Jupyter Notebook, Visual Studio Code

### 분석방법
- 기상청과 국립환경과학원의 OPEN API으로 수집한 자료 활용
- Exploratory Data Analysis (EDA) 와 Feature Importance로 피쳐 간의 관계 분석과 영향력 확인
- 머신러닝 : 회귀와 분류 알고리즘 비교
- 딥러닝 : CNN 기반 모델 생성 후 하이퍼 파라미터 튜닝으로 예측력 최대화
- 정확도(Accuracy) 를 지표로 설정, 모든 모델의 예측력 평가 후 서비스를 위한 모델 선정

### 분석결과
- 종로구 익일 미세먼지 농도 예측에 가장 큰 영향력을 미치는 요소는 전날 이슬점 온도, 미세먼지 농도, 일산화탄소 농도, 기온, 증기압 순으로 높았음
- 종로구 익일 미세먼지 농도 예측에 가장 큰 영향력을 미치는 위치는 경기도 파주, 경기도 포천, 경기도 하남, 경기도 동두천, 경기도 양평 순으로 높았음으로 
  서울시 주변 경기도의 요소들이 크게 영향을 미치는 것으로 확인되었음
- 기상요소 중,  현상번호는 백령도, 인천 중구, 파주, 강변북로의 비, 눈, 안개, 황사 등이 종로구 익일 미세먼지 농도에 영향을 미치고 있었음
- 스태킹을 이용하여 회귀 모델과 분류 모델을 앙상블한 머신러닝 모델이 가장 정확도가 높았음

### 웹서비스
https://seoyoonjoannechang.github.io/

### [Trello](https://trello.com/b/PWqvjjF5/shinchon-ultra-machine-learning-team)
- 프로젝트 기간동안 각 팀원들 업무 진행 상황 공유   
