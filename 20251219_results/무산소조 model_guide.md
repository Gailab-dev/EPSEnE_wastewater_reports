# 04 무산소조 예측 모델 가이드

**작성일**: 2025-12-19
**모델 ID**: REQ004
**버전**: v1.0

---

## 📋 목차

1. [모델 개요](#모델-개요)
2. [모델 알고리즘 및 구조](#모델-알고리즘-및-구조)
3. [입력 데이터 정의](#입력-데이터-정의)
4. [입력 데이터 전처리 과정](#입력-데이터-전처리-과정)
5. [출력 데이터 정의](#출력-데이터-정의)
6. [모델 사용 방법](#모델-사용-방법)
7. [성능 지표](#성능-지표)

---

## 모델 개요

### 목적
하수처리시설 **무산소조(Anoxic Tank)의 운영 상태를 예측**하여 질소 제거 효율 최적화 및 탈질 공정 안정화를 지원합니다.

### 핵심 특징
- **Multi-Output 예측**: 무산소조 주요 지표를 단일 모델로 동시 예측
- **시계열 데이터**: 시간당 측정 데이터 기반 예측
- **실시간 운영 지원**: 1시간 후 무산소조 상태 예측 가능
- **도메인 지식 활용**: 탈질 공정 특성을 반영한 피처 엔지니어링

### 주요 사양
| 항목 | 상세 |
|------|------|
| **모델 타입** | XGBoost Multi-Output Regression |
| **예측 대상** | 무산소조 NH4-N, NO3-N, MLSS |
| **예측 시점** | t 시점 데이터로 t+1 시점 예측 (1시간 후) |
| **입력 피처** | 약 50개 (유입수 + 혐기조 + 무산소조 운전 데이터 + 시간 피처 + 도메인 피처) |
| **학습 데이터** | 2024년 시간당 데이터 (8,765개 샘플) |
| **목표 성능** | R² ≥ 0.85, MAPE ≤ 15% |

---

## 모델 알고리즘 및 구조

### 1. 알고리즘: XGBoost Multi-Output Regression

#### XGBoost (eXtreme Gradient Boosting)
- **타입**: Gradient Boosting Decision Tree 기반 앙상블 모델
- **특징**:
  - 높은 예측 정확도
  - 병렬 처리로 빠른 학습 속도
  - 정규화 기능으로 과적합 방지
  - 결측치 자동 처리

#### Multi-Output Wrapper
- `sklearn.multioutput.MultiOutputRegressor` 사용
- 3개의 독립적인 XGBoost 모델을 내부에서 학습
- 각 출력(NH4-N, NO3-N, MLSS)마다 별도의 예측기 생성

### 2. 모델 구조

```
입력 데이터 (약 50개 피처)
    ↓
[전처리 레이어]
    ├─ 이상치 클리핑 (1%~99% 분위수)
    ├─ 시간 피처 생성
    ├─ 도메인 피처 생성 (C/N 비율, 질소 부하 등)
    ├─ Lag 피처 생성 (24h, 168h)
    └─ Rolling 통계 생성 (24h 윈도우)
    ↓
[정규화 레이어]
    └─ StandardScaler (평균 0, 표준편차 1)
    ↓
[XGBoost Multi-Output]
    ├─ Estimator #1: 무산소조 NH4-N 예측
    ├─ Estimator #2: 무산소조 NO3-N 예측
    └─ Estimator #3: 무산소조 MLSS 예측
    ↓
[역정규화 레이어]
    └─ 예측값을 원래 스케일로 복원
    ↓
출력 데이터 (3개 지표)
```

### 3. 하이퍼파라미터 (권장값)

```python
xgb_params = {
    'n_estimators': 200,         # 트리 개수
    'max_depth': 4,              # 트리 최대 깊이 (과적합 방지)
    'learning_rate': 0.05,       # 학습률 (천천히 학습)
    'min_child_weight': 1,       # 리프 노드 최소 가중치
    'subsample': 0.8,            # 행 샘플링 비율 (80%)
    'colsample_bytree': 0.8,     # 열 샘플링 비율 (80%)
    'gamma': 0,                  # 분할 최소 손실 감소
    'reg_alpha': 0.1,            # L1 정규화 (Lasso)
    'reg_lambda': 1.0,           # L2 정규화 (Ridge)
    'random_state': 42,          # 재현성을 위한 시드
    'n_jobs': -1                 # 모든 CPU 코어 사용
}
```

### 4. 검증 전략

#### Time Series Cross-Validation (5-Fold)
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Fold 1: Train [0:1753], Test [1753:3506]
# Fold 2: Train [0:3506], Test [3506:5259]
# Fold 3: Train [0:5259], Test [5259:7012]
# Fold 4: Train [0:7012], Test [7012:8765]
# Fold 5: Train [0:8765], Test [8765:10518]
```

**목적**: 시계열 데이터의 시간 순서를 보존하며 모델의 일반화 성능 검증

---

## 입력 데이터 정의

### 데이터 소스
- **파일**: `dataset/20251024/무산소조.csv`
- **기간**: 2024년 1월 1일 ~ 12월 31일
- **샘플 수**: 8,765개 (시간당 데이터)
- **총 변수 수**: 약 50개

### 1. 원본 입력 변수

#### 유입수 관련 변수 (6개)
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `유입유량` | m³/일 | 하수처리장 유입 유량 | 50,000~200,000 |
| `유입BOD` | mg/L | 생물화학적 산소요구량 | 50~300 |
| `유입TN` | mg/L | 총 질소 농도 | 10~50 |
| `유입TP` | mg/L | 총 인 농도 | 1~10 |
| `유입TOC` | mg/L | 총 유기탄소 농도 | 20~100 |
| `유입SS` | mg/L | 부유물질 농도 | 30~300 |

#### 혐기조 관련 변수 (3개)
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `혐기_BOD` | mg/L | 혐기조 BOD 농도 | 10~100 |
| `혐기_Po4P` | mg/L | 혐기조 인산염 인 | 0.5~5 |
| `혐기_MLSS` | mg/L | 혐기조 부유고형물 농도 | 1,500~3,500 |

#### 무산소조 운전 데이터 (4개)
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `무산소_BOD` | mg/L | 무산소조 BOD 농도 (lag) | 10~80 |
| `무산소_NH4` | mg/L | 암모니아성 질소 (lag) | 0.5~10 |
| `무산소_NO3` | mg/L | 질산성 질소 (lag) | 10~30 |
| `무산소_MLSS` | mg/L | 부유고형물 농도 (lag) | 1,500~3,500 |

#### 기타 변수 (2개)
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `수온` | ℃ | 수온 | 5~30 |
| `pH` | - | pH | 6~9 |

### 2. 시간 피처 (4개)

| 변수명 | 타입 | 설명 | 범위 |
|--------|------|------|------|
| `month` | int | 월 (1~12월) | 1~12 |
| `day_of_week` | int | 요일 (0=월요일, 6=일요일) | 0~6 |
| `is_weekend` | int | 주말 여부 (0=평일, 1=주말) | 0, 1 |
| `season` | int | 계절 (1=봄, 2=여름, 3=가을, 4=겨울) | 1~4 |

**※ 제외**: `hour`, `day` 변수 (과적합 방지)

### 3. 도메인 기반 피처

#### 3.1 비율 피처 (탈질 효율 관련)
| 변수명 | 계산식 | 의미 | 활용 |
|--------|--------|------|------|
| `BOD_TN_ratio` | BOD / (TN + ε) | C/N 비율 | 탈질 효율 예측 |
| `NO3_NH4_ratio` | NO3 / (NH4 + ε) | 질산화 진행도 | 탈질 기질 농도 |
| `TN_유량_ratio` | TN × 유량 / 1e6 | 질소 부하량 | 탈질 부하 평가 |
| `MLSS_ratio` | 무산소_MLSS / 혐기_MLSS | 슬러지 농도 비율 | 미생물 활성도 |

**※ ε = 1e-10**: 0으로 나누기 방지

#### 3.2 변화율 피처
| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `TN_변화율` | (TNₜ - TNₜ₋₁) / TNₜ₋₁ | 시간당 TN 변화율 |
| `NH4_변화율` | (NH4ₜ - NH4ₜ₋₁) / NH4ₜ₋₁ | 시간당 NH4 변화율 |
| `NO3_변화율` | (NO3ₜ - NO3ₜ₋₁) / NO3ₜ₋₁ | 시간당 NO3 변화율 |

#### 3.3 부하 피처
| 변수명 | 계산식 | 단위 | 의미 |
|--------|--------|------|------|
| `TN_부하량` | 유입유량 × 유입TN × 1e-3 | kg/일 | 질소 부하량 |
| `BOD_부하량` | 유입유량 × 유입BOD × 1e-3 | kg/일 | 유기물 부하량 |

### 4. Lag 피처

**목적**: 과거 데이터 패턴 학습

| 시간 범위 | 변수 개수 | 변수 예시 |
|-----------|-----------|-----------|
| 24시간 전 (1일) | 약 8개 | `유입TN_lag24`, `무산소_NH4_lag24` |
| 168시간 전 (1주) | 약 8개 | `유입TN_lag168`, `무산소_NO3_lag168` |

**※ 개선사항**: 1시간 Lag 제거 (실시간 예측 불가능한 정보)

### 5. Rolling 통계 피처

**목적**: 최근 24시간 트렌드 파악

| 통계량 | 변수 예시 |
|--------|-----------|
| 평균 (mean) | `유입TN_rolling_mean24` |
| 표준편차 (std) | `유입TN_rolling_std24` |
| 최대값 (max) | `무산소_NH4_rolling_max24` |
| 최소값 (min) | `무산소_NO3_rolling_min24` |

**대상 변수**: 유입TN, 유입BOD, 무산소_NH4, 무산소_NO3

---

## 입력 데이터 전처리 과정

### 전처리 파이프라인

```
원본 데이터 (CSV)
    ↓
[1단계] 데이터 로드 및 정제
    ├─ CSV 파일 읽기
    ├─ 인덱스 재설정
    ├─ 초기 결측치 제거
    └─ 데이터 타입 변환 (float)
    ↓
[2단계] 이상치 처리
    ├─ 방법: 분위수 기반 클리핑 (1%~99%)
    └─ 적용 대상: 모든 수치형 변수
    ↓
[3단계] 시간 피처 생성
    ├─ month, day_of_week 추출
    ├─ is_weekend 생성
    └─ season 생성
    ↓
[4단계] 도메인 피처 생성
    ├─ 비율 피처 4개
    ├─ 변화율 피처 3개
    └─ 부하 피처 2개
    ↓
[5단계] Lag 피처 생성
    ├─ 24h Lag (1일)
    └─ 168h Lag (1주)
    ↓
[6단계] Rolling 통계 생성
    ├─ 24h 윈도우
    └─ mean, std, max, min
    ↓
[7단계] 결측치 처리
    ├─ Lag/Rolling으로 인한 결측 제거
    └─ 초기 168시간 데이터 제거
    ↓
[8단계] Feature/Target 분리
    ├─ X: 입력 피처
    └─ y: 출력 변수 (NH4, NO3, MLSS)
    ↓
[9단계] Train/Test 분할
    ├─ 시계열 순차 분할 (80:20)
    └─ 시간 순서 유지
    ↓
[10단계] 정규화
    ├─ StandardScaler (X)
    └─ StandardScaler (y)
    ↓
[11단계] 모델 학습
```

### 주요 전처리 코드

#### 1단계: 데이터 로드
```python
df = pd.read_csv('../dataset/20251024/무산소조.csv', header=0)
df = df.reset_index(drop=True)
df.dropna(inplace=True)
df[df.columns[1:]] = df[df.columns[1:]].astype(float)
```

#### 2단계: 이상치 처리
```python
def clip_outliers(df, columns, lower_quantile=0.01, upper_quantile=0.99):
    df_clipped = df.copy()
    for col in columns:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        df_clipped[col] = df[col].clip(lower, upper)
    return df_clipped

numeric_cols = ['유입유량', '유입BOD', '유입TN', '유입TP', '유입TOC', '유입SS',
                '혐기_BOD', '혐기_Po4P', '혐기_MLSS',
                '무산소_BOD', '무산소_NH4', '무산소_NO3', '무산소_MLSS',
                '수온', 'pH']
df = clip_outliers(df, numeric_cols)
```

#### 3단계: 시간 피처 생성
```python
df['측정일시'] = pd.to_datetime(df['측정일시'])
df['month'] = df['측정일시'].dt.month
df['day_of_week'] = df['측정일시'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

def get_season(month):
    if month in [3, 4, 5]: return 1  # 봄
    elif month in [6, 7, 8]: return 2  # 여름
    elif month in [9, 10, 11]: return 3  # 가을
    else: return 4  # 겨울

df['season'] = df['month'].apply(get_season)
```

#### 10단계: 정규화
```python
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
```

---

## 출력 데이터 정의

### 1. 예측 대상 (3개 지표)

**무산소조 탈질 성능 지표**:

| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `무산소_NH4` | mg/L | 암모니아성 질소 농도 | 0.5~10 |
| `무산소_NO3` | mg/L | 질산성 질소 농도 | 10~30 |
| `무산소_MLSS` | mg/L | 부유고형물 농도 | 1,500~3,500 |

**참고**:
- NH4-N: 탈질 후 남은 암모니아 농도 (낮을수록 좋음)
- NO3-N: 탈질 기질 농도 (호기조에서 생성)
- MLSS: 미생물 농도 (탈질 효율과 관련)

### 2. 출력 형식

#### Python (numpy array)
```python
# 단일 샘플 예측
y_pred = model.predict(X_new)
# 출력: [[NH4, NO3, MLSS]]

# 다중 샘플 예측
y_pred = model.predict(X_new_batch)
# 출력: [[NH4_1, NO3_1, MLSS_1],
#        [NH4_2, NO3_2, MLSS_2], ...]
```

#### JSON 형식
```json
{
  "timestamp": "2024-12-31 12:00:00",
  "predictions": {
    "무산소_NH4": 6.49,
    "무산소_NO3": 21.05,
    "무산소_MLSS": 2439.0
  },
  "model_version": "REQ004_v1.0"
}
```

---

## 모델 사용 방법

### 1. 기본 예측

```python
import joblib
import pandas as pd

# 모델 로드
model = joblib.load('04 무산소조/models/anoxic_model_20251219.pkl')
scaler_X = joblib.load('04 무산소조/models/scaler_X_20251219.pkl')
scaler_y = joblib.load('04 무산소조/models/scaler_y_20251219.pkl')
feature_cols = joblib.load('04 무산소조/models/feature_cols_20251219.pkl')

# 새로운 데이터 준비 (1시간 전 데이터)
X_new = ...  # DataFrame 형태

# 피처 정렬
X_new_aligned = X_new[feature_cols]

# 정규화
X_new_scaled = scaler_X.transform(X_new_aligned)

# 예측
y_pred_scaled = model.predict(X_new_scaled)

# 역정규화
y_pred = scaler_y.inverse_transform(y_pred_scaled)

print("예측 결과:")
print(f"무산소_NH4: {y_pred[0][0]:.2f} mg/L")
print(f"무산소_NO3: {y_pred[0][1]:.2f} mg/L")
print(f"무산소_MLSS: {y_pred[0][2]:.0f} mg/L")
```

### 2. 배치 예측

```python
# 여러 시점 동시 예측
X_batch = pd.read_csv('new_data_batch.csv')
X_batch_aligned = X_batch[feature_cols]
X_batch_scaled = scaler_X.transform(X_batch_aligned)

y_batch_scaled = model.predict(X_batch_scaled)
y_batch = scaler_y.inverse_transform(y_batch_scaled)

# 결과 저장
results = pd.DataFrame(y_batch, columns=['무산소_NH4', '무산소_NO3', '무산소_MLSS'])
results.to_csv('predictions.csv', index=False)
```

### 3. 실시간 운영 함수

```python
def handle_missing_realtime(df, limit=3):
    """결측치 처리: 최대 3시간까지 전방향 채우기"""
    return df.fillna(method='ffill', limit=limit)

def predict_with_fallback(model, X, y_train_history, min_lag_hours=168):
    """Cold Start 대응: Lag 피처 부족 시 평균값 사용"""
    if hasattr(X, 'columns'):
        lag_cols = [col for col in X.columns if 'lag' in col]
        if X[lag_cols].isnull().any().any():
            print("⚠️ Cold start detected - using historical average")
            return y_train_history[-min_lag_hours:].mean(axis=0)
    return model.predict(X)

def monitor_model_performance(y_true, y_pred, threshold_mape=20):
    """성능 모니터링: MAPE 20% 초과 시 재학습 필요"""
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    if mape > threshold_mape:
        print(f"⚠️ Model degradation: MAPE {mape:.2f}% > {threshold_mape}%")
        return True
    return False
```

---

## 성능 지표

### 1. 목표 성능

| 지표 | 목표값 | 설명 |
|------|--------|------|
| **R² Score** | ≥ 0.85 | 결정계수 (1에 가까울수록 좋음) |
| **MAPE** | ≤ 15% | 평균 절대 백분율 오차 |
| **RMSE** | 확정 예정 | 평균 제곱근 오차 |

### 2. 평가 지표 계산

```python
# Safe MAPE (0 나누기 방지)
def safe_mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Symmetric MAPE
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)
```

### 3. Time Series Cross-Validation

5-Fold CV 평균 성능 목표:

```
무산소_NH4:  R² = 0.85 ± 0.05
무산소_NO3:  R² = 0.85 ± 0.05
무산소_MLSS: R² = 0.85 ± 0.05
```

### 4. 성능 모니터링 기준

#### 정기 재학습 주기
- **기본**: 30일마다 재학습
- **이유**: 데이터 분포 변화 대응

#### 조기 재학습 기준
- **MAPE > 20%**: 모델 성능 저하 감지 시 즉시 재학습
- **연속 3일 이상 예측 오차 증가**: 트렌드 변화 감지

---

## 참고 문서

- **01 유입수/req002_model.ipynb**: 유입 부하량 예측 참조
- **03 혐기조/model.ipynb**: 혐기조 예측 참조
- **06 이차침전지/model.ipynb**: 성능 검증 참조
- **prompt.md**: 모델 개발 가이드라인
