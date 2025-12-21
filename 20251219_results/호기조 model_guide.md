# REQ005 호기조 예측 모델 가이드

**작성일**: 2025-12-17
**모델 ID**: REQ005
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
하수처리시설 **호기조(Aerobic Tank)의 운영 상태를 예측**하여 질소 제거 효율 최적화 및 공정 안정화를 지원합니다.

### 핵심 특징
- **Multi-Output 예측**: 호기조 주요 지표를 단일 모델로 동시 예측
- **일별 데이터**: 일별 측정 데이터 기반 예측
- **실시간 운영 지원**: 다음 날 호기조 상태 예측 가능
- **도메인 지식 활용**: 호기조 공정 특성을 반영한 피처 엔지니어링

### 주요 사양
| 항목 | 상세 |
|------|------|
| **모델 타입** | XGBoost Multi-Output Regression |
| **예측 대상** | 호기조 DO, 호기조 NH4_N, 호기조 NO3_N |
| **예측 시점** | t 시점 데이터로 t+1 시점 예측 (1일 후) |
| **입력 피처** | 약 50개 (원본 변수 + 시간 피처 + 도메인 피처 + Lag 피처 + Rolling 통계) |
| **학습 데이터** | 2012-2024년 일별 데이터 (약 4,500개 샘플) |
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
- 각 출력(DO, NH4_N, NO3_N)마다 별도의 예측기 생성

### 2. 모델 구조

```
입력 데이터 (약 50개 피처)
    ↓
[전처리 레이어]
    ├─ 이상치 클리핑 (1%~99% 분위수)
    ├─ 시간 피처 생성 (월, 요일, 계절)
    ├─ 도메인 피처 생성 (질산화 효율, C/N 비율)
    ├─ Lag 피처 생성 (7일, 30일)
    └─ Rolling 통계 생성 (7일 윈도우)
    ↓
[정규화 레이어]
    └─ StandardScaler (평균 0, 표준편차 1)
    ↓
[XGBoost Multi-Output]
    ├─ Estimator #1: 호기조 DO 예측
    ├─ Estimator #2: 호기조 NH4_N 예측
    └─ Estimator #3: 호기조 NO3_N 예측
    ↓
[역정규화 레이어]
    └─ 예측값을 원래 스케일로 복원
    ↓
출력 데이터 (3개 지표)
```

### 3. 하이퍼파라미터 (최적화 완료)

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
    'n_jobs': -1                 # 병렬 처리 (모든 CPU 사용)
}
```

### 4. 검증 전략

#### Time Series Cross-Validation (5-Fold)
```
Fold 1: [Train ████████      ] [Val ██]
Fold 2: [Train ██████████    ] [Val ██]
Fold 3: [Train ████████████  ] [Val ██]
Fold 4: [Train ██████████████] [Val ██]
Fold 5: [Train ████████████████] [Val ██]
```

- **목적**: 시계열 데이터의 시간 순서 유지
- **장점**: 모델 안정성 및 일반화 성능 검증
- **결과**: 각 타겟별 R² Mean ± Std 산출

---

## 입력 데이터 정의

### 1. 원본 입력 변수 (16개)

| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `일자` | date | 측정 일자 | - |
| `요일` | str | 요일 | 월~일 |
| `날씨` | str | 날씨 정보 | 맑음, 흐림, 비, 눈 |
| `기온` | °C | 대기 온도 | -20~40 |
| `pH` | - | 호기조 pH | 6.5~8.5 |
| `수온` | °C | 호기조 수온 | 5~30 |
| `DO` | mg/L | 용존산소 농도 (입력) | 1.0~4.0 |
| `HRT` | hr | 수리학적 체류시간 | 3~8 |
| `MLSS` | mg/L | 혼합액부유물질 농도 | 2000~5000 |
| `ASRT` | 일 | 슬러지 체류시간 | 5~20 |
| `외부반송량` | m³/일 | 외부 슬러지 반송량 | 10000~25000 |
| `내부반송률` | % | 내부 반송 비율 | 100~300 |
| `SV` | % | 슬러지 침강률 | 20~50 |
| `SVI` | mL/g | 슬러지 용적지수 | 80~200 |
| `송풍량` | m³/h | 공기 공급량 (시간당) | 4000~10000 |
| `송풍량(일)` | m³/일 | 공기 공급량 (일별) | 96000~240000 |

### 2. 질소/인 농도 변수 (8개)

#### 유입 농도
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `NH4_N` | mg/L | 암모니아성 질소 (유입) | 10~50 |
| `NO2_N` | mg/L | 아질산성 질소 (유입) | 0~5 |
| `NO3_N` | mg/L | 질산성 질소 (유입) | 0~10 |
| `PO4_P` | mg/L | 인산염 (유입) | 2~10 |

#### 유출 농도
| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `NH4_N.1` | mg/L | 암모니아성 질소 (유출) | 0~5 |
| `NO2_N.1` | mg/L | 아질산성 질소 (유출) | 0~2 |
| `NO3_N.1` | mg/L | 질산성 질소 (유출) | 5~15 |
| `PO4_P.1` | mg/L | 인산염 (유출) | 0~2 |

### 3. 시간 피처 (5개)

| 변수명 | 타입 | 설명 | 범위 |
|--------|------|------|------|
| `month` | int | 월 (1~12월) | 1~12 |
| `day_of_week` | int | 요일 (0=월요일, 6=일요일) | 0~6 |
| `is_weekend` | int | 주말 여부 (0=평일, 1=주말) | 0, 1 |
| `season` | int | 계절 (1=봄, 2=여름, 3=가을, 4=겨울) | 1~4 |
| `day` | int | 일자 (1~31일) | 1~31 |

**※ 주의**: `day` 변수는 모델 학습 시 제외됨 (과적합 방지)

### 4. 도메인 기반 피처 (8개)

#### 4.1 질산화 효율 피처 (3개)
| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `nitrification_efficiency` | (NH4_N - NH4_N.1) / (NH4_N + ε) | 질산화 효율 |
| `denitrification_potential` | NO3_N.1 / (NO3_N + ε) | 탈질 잠재력 |
| `total_N_removal` | (TN_in - TN_out) / (TN_in + ε) | 총 질소 제거율 |

**※ ε = 1e-10**: 0으로 나누기 방지

#### 4.2 비율 피처 (3개)
| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `MLSS_SVI_ratio` | MLSS / (SVI + ε) | 슬러지 품질 지표 |
| `DO_수온_ratio` | DO / (수온 + ε) | 온도 보정 DO |
| `송풍량_MLSS_ratio` | 송풍량(일) / (MLSS + ε) | 공기공급 효율 |

#### 4.3 변화율 피처 (2개)
| 변수명 | 계산식 | 의미 |
|--------|--------|------|
| `DO_변화율` | (DOₜ - DOₜ₋₁) / (DOₜ₋₁ + ε) | 일별 DO 변화율 |
| `MLSS_변화율` | (MLSSₜ - MLSSₜ₋₁) / (MLSSₜ₋₁ + ε) | 일별 MLSS 변화율 |

### 5. Lag 피처 (12개)

**목적**: 과거 데이터 패턴 학습

| 시간 범위 | 변수 개수 | 변수 예시 |
|-----------|-----------|-----------|
| 7일 전 | 6개 | `DO_lag7`, `MLSS_lag7`, `NH4_N_lag7`, ... |
| 30일 전 | 6개 | `DO_lag30`, `MLSS_lag30`, `NH4_N_lag30`, ... |

**대상 변수**: DO, MLSS, NH4_N, NO3_N, 송풍량, HRT

### 6. Rolling 통계 피처 (16개)

**목적**: 최근 7일 트렌드 파악

| 통계량 | 변수 개수 | 변수 예시 |
|--------|-----------|-----------|
| 평균 (mean) | 4개 | `DO_rolling_mean7` |
| 표준편차 (std) | 4개 | `DO_rolling_std7` |
| 최대값 (max) | 4개 | `DO_rolling_max7` |
| 최소값 (min) | 4개 | `DO_rolling_min7` |

**대상 변수**: DO, MLSS, NH4_N, NO3_N

### 7. 전체 피처 구성 요약

```
총 약 50개 피처
├─ 원본 변수: 24개 (운전 변수 16개 + 질소/인 8개)
├─ 시간 피처: 4개 (month, day_of_week, is_weekend, season)
├─ 도메인 피처: 8개 (질산화 효율 3개 + 비율 3개 + 변화율 2개)
├─ Lag 피처: 12개 (7일 × 6개 + 30일 × 6개)
├─ Rolling 피처: 16개 (4개 통계량 × 4개 변수)
└─ 제외: day (과적합 방지)
```

---

## 입력 데이터 전처리 과정

### 전처리 파이프라인

```
원본 데이터 (CSV)
    ↓
[1단계] 데이터 로드 및 정제
    ├─ 헤더 행 읽기
    ├─ 인덱스 재설정
    ├─ 초기 결측치 제거
    └─ 데이터 타입 변환 (float, datetime)
    ↓
[2단계] 이상치 처리
    ├─ 방법: 분위수 기반 클리핑
    ├─ 범위: 1% ~ 99% 분위수
    └─ 대상: 모든 수치형 변수
    ↓
[3단계] 시간 피처 생성
    ├─ datetime 변환
    ├─ month, day_of_week 추출
    ├─ is_weekend 생성
    └─ season 생성 (월 → 계절 매핑)
    ↓
[4단계] 도메인 피처 생성
    ├─ 질산화 효율 피처 (3개)
    ├─ 비율 피처 (3개)
    └─ 변화율 피처 (2개)
    ↓
[5단계] 타겟 변수 정의
    ├─ 호기조_DO = DO (유출 시점)
    ├─ 호기조_NH4_N = NH4_N.1
    └─ 호기조_NO3_N = NO3_N.1
    ↓
[6단계] Lag 피처 생성
    ├─ 7일 Lag (6개 변수)
    └─ 30일 Lag (6개 변수)
    ↓
[7단계] Rolling 통계 생성
    ├─ 윈도우: 7일
    ├─ 통계량: mean, std, max, min
    └─ 대상: 4개 변수
    ↓
[8단계] 결측치 처리
    ├─ 초기 30일 제거 (Lag 피처 생성으로 인한 결측)
    └─ 나머지 결측치 제거 (dropna)
    ↓
[9단계] Feature/Target 분리
    ├─ 제외 컬럼: 일자, 타겟 3개, day
    └─ Feature: 약 50개, Target: 3개
    ↓
[10단계] Train/Test 분할
    ├─ 방법: 시계열 순차 분할
    ├─ 비율: Train 80% / Test 20%
    └─ Shuffle: False (시간 순서 유지)
    ↓
[11단계] 정규화
    ├─ 방법: StandardScaler
    ├─ 공식: (X - μ) / σ
    ├─ 적용: X_train, y_train으로 fit
    └─ 변환: X_test, y_test는 transform만
    ↓
학습 준비 완료
```

### 각 단계 상세 설명

#### 1단계: 데이터 로드 및 정제
```python
df = pd.read_csv('../dataset/호기조.csv')
df['일자'] = pd.to_datetime(df['일자'])
df = df.dropna(subset=['일자'])
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].astype(float)
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

numeric_cols = ['pH', '수온(oC)', 'DO(mg/L)', 'MLSS(mg/L)', 'HRT', 'ASRT(일)',
                '송풍량(m3/h)', 'NH4_N', 'NO3_N', 'PO4_P']
df = clip_outliers(df, numeric_cols, lower_quantile=0.01, upper_quantile=0.99)
```

#### 3단계: 시간 피처 생성
```python
df['month'] = df['일자'].dt.month
df['day_of_week'] = df['일자'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

def get_season(month):
    if month in [3, 4, 5]: return 1  # 봄
    elif month in [6, 7, 8]: return 2  # 여름
    elif month in [9, 10, 11]: return 3  # 가을
    else: return 4  # 겨울

df['season'] = df['month'].apply(get_season)
```

#### 4단계: 도메인 피처 생성
```python
# 질산화 효율 피처
df['nitrification_efficiency'] = (df['NH4_N'] - df['NH4_N.1']) / (df['NH4_N'] + 1e-10)
df['denitrification_potential'] = df['NO3_N.1'] / (df['NO3_N'] + 1e-10)

# 비율 피처
df['MLSS_SVI_ratio'] = df['MLSS(mg/L)'] / (df['SVI'] + 1e-10)
df['DO_수온_ratio'] = df['DO(mg/L)'] / (df['수온(oC)'] + 1e-10)
df['송풍량_MLSS_ratio'] = df['송풍량(m3/d)'] / (df['MLSS(mg/L)'] + 1e-10)

# 변화율 피처
df['DO_변화율'] = df['DO(mg/L)'].pct_change()
df['MLSS_변화율'] = df['MLSS(mg/L)'].pct_change()
```

#### 5단계: 타겟 변수 정의
```python
# 호기조 유출 상태 예측
df['호기조_DO'] = df['DO(mg/L)']
df['호기조_NH4_N'] = df['NH4_N.1']
df['호기조_NO3_N'] = df['NO3_N.1']
```

#### 6단계: Lag 피처 생성
```python
lag_features = ['DO(mg/L)', 'MLSS(mg/L)', 'NH4_N', 'NO3_N', '송풍량(m3/h)', 'HRT']
lag_periods = [7, 30]  # 1일 Lag 제거 (실시간 예측 고려)

for feature in lag_features:
    for lag in lag_periods:
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
```

#### 7단계: Rolling 통계 생성
```python
rolling_features = ['DO(mg/L)', 'MLSS(mg/L)', 'NH4_N', 'NO3_N']
window = 7

for feature in rolling_features:
    df[f'{feature}_rolling_mean7'] = df[feature].rolling(window=window).mean()
    df[f'{feature}_rolling_std7'] = df[feature].rolling(window=window).std()
    df[f'{feature}_rolling_max7'] = df[feature].rolling(window=window).max()
    df[f'{feature}_rolling_min7'] = df[feature].rolling(window=window).min()
```

#### 8단계: 결측치 처리
```python
df_clean = df.iloc[30:].copy()  # 초기 30일 제거
df_clean = df_clean.dropna()      # 나머지 결측치 제거
```

#### 9단계: Feature/Target 분리
```python
target_cols = ['호기조_DO', '호기조_NH4_N', '호기조_NO3_N']
exclude_cols = ['일자', '호기조_DO', '호기조_NH4_N', '호기조_NO3_N', 'day']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

X = df_clean[feature_cols]  # 약 50개 피처
y = df_clean[target_cols]    # 3개 타겟
```

#### 10단계: Train/Test 분할
```python
split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]
```

#### 11단계: 정규화
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

### 1. 예측 대상 (3개)

| 변수명 | 단위 | 설명 | 정상 범위 |
|--------|------|------|-----------|
| `호기조_DO` | mg/L | 호기조 용존산소 농도 | 1.5~4.0 |
| `호기조_NH4_N` | mg/L | 호기조 유출 암모니아성 질소 | 0~5 |
| `호기조_NO3_N` | mg/L | 호기조 유출 질산성 질소 | 5~15 |

### 2. 출력 형식

#### Python (numpy array)
```python
# 단일 샘플 예측
y_pred = model.predict(X_new)
# 출력: [[2.5, 1.2, 8.5]]
#        [DO, NH4_N, NO3_N]

# 다중 샘플 예측
y_pred = model.predict(X_new_batch)
# 출력:
# [[2.5, 1.2, 8.5],
#  [2.3, 1.5, 9.2],
#  [2.8, 0.9, 7.8]]
```

#### JSON 형식
```json
{
  "date": "2024-12-31",
  "predictions": {
    "호기조_DO": 2.5,
    "호기조_NH4_N": 1.2,
    "호기조_NO3_N": 8.5
  },
  "unit": {
    "DO": "mg/L",
    "NH4_N": "mg/L",
    "NO3_N": "mg/L"
  },
  "model_version": "REQ005_v1.0"
}
```

### 3. 출력 데이터 해석

#### 호기조 DO (Dissolved Oxygen)
- **의미**: 호기조 내 용존산소 농도
- **활용**:
  - 질산화 효율 모니터링
  - 송풍량 조절 기준
  - 에너지 효율 최적화

#### 호기조 NH4_N (Ammonium Nitrogen)
- **의미**: 호기조 유출수의 암모니아성 질소 농도
- **활용**:
  - 질산화 완료 여부 확인
  - 방류수 수질 기준 준수 여부
  - HRT 조절 필요성 판단

#### 호기조 NO3_N (Nitrate Nitrogen)
- **의미**: 호기조 유출수의 질산성 질소 농도
- **활용**:
  - 질산화 성공 지표
  - 탈질조 운영 조건 설정
  - 내부반송률 조절

### 4. 예측 신뢰도 지표

모델과 함께 다음 지표를 제공하여 예측 신뢰도를 평가할 수 있습니다:

```python
# 예측 구간 (Prediction Interval)
y_pred_lower = y_pred - 1.96 * y_pred_std  # 95% 하한
y_pred_upper = y_pred + 1.96 * y_pred_std  # 95% 상한

# 출력 예시
{
  "호기조_DO": {
    "prediction": 2.5,
    "lower_95": 2.2,
    "upper_95": 2.8,
    "confidence": 0.95
  }
}
```

---

## 모델 사용 방법

### 1. 모델 로드

```python
import joblib
import pandas as pd
import numpy as np

# 저장된 모델 및 스케일러 로드
model = joblib.load('aerobic_tank_model_YYYYMMDD_HHMMSS.pkl')
scaler_X = joblib.load('scaler_X_YYYYMMDD_HHMMSS.pkl')
scaler_y = joblib.load('scaler_y_YYYYMMDD_HHMMSS.pkl')
feature_cols = joblib.load('feature_cols_YYYYMMDD_HHMMSS.pkl')

print(f"✓ 모델 로드 완료")
print(f"  - 피처 수: {len(feature_cols)}개")
print(f"  - 출력 수: 3개 (DO, NH4_N, NO3_N)")
```

### 2. 새로운 데이터 예측 (기본)

```python
# 1. 새로운 데이터 준비 (DataFrame 형식)
X_new = pd.DataFrame({
    'pH': [7.0],
    '수온(oC)': [15.5],
    'DO(mg/L)': [2.5],
    # ... (모든 약 50개 피처 필요)
})

# 2. 피처 순서 맞추기
X_new_aligned = X_new[feature_cols]

# 3. 정규화
X_new_scaled = scaler_X.transform(X_new_aligned)

# 4. 예측
y_pred_scaled = model.predict(X_new_scaled)

# 5. 역정규화
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 6. 결과 출력
print(f"예측 결과:")
print(f"  호기조 DO: {y_pred[0][0]:.2f} mg/L")
print(f"  호기조 NH4_N: {y_pred[0][1]:.2f} mg/L")
print(f"  호기조 NO3_N: {y_pred[0][2]:.2f} mg/L")
```

### 3. 실시간 운영 시나리오

```python
# 실시간 운영 함수 정의
def handle_missing_realtime(df, limit=3):
    """결측치 처리: 최대 3일까지 전방향 채우기"""
    return df.fillna(method='ffill', limit=limit)

def predict_with_fallback(model, X, y_train_history, min_lag_days=30):
    """Cold Start 대응: Lag 피처 부족 시 평균값 사용"""
    if hasattr(X, 'columns'):
        lag_cols = [col for col in X.columns if 'lag' in col]
        if X[lag_cols].isnull().any().any():
            print("⚠️ Cold start detected - using historical average")
            return y_train_history[-min_lag_days:].mean(axis=0)
    return model.predict(X)

def monitor_model_performance(y_true, y_pred, threshold_mape=20):
    """성능 모니터링: MAPE 20% 초과 시 재학습 필요"""
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    if mape > threshold_mape:
        print(f"⚠️ Model degradation: MAPE {mape:.2f}% > {threshold_mape}%")
        return True
    return False

# 실시간 예측 워크플로우
# 1. 새로운 데이터 수신
X_new = get_latest_data()  # 실시간 데이터 수신

# 2. 결측치 처리
X_new_filled = handle_missing_realtime(X_new, limit=3)

# 3. 피처 정렬 및 정규화
X_new_aligned = X_new_filled[feature_cols]
X_new_scaled = scaler_X.transform(X_new_aligned)

# 4. 예측 (Cold Start 대응)
y_pred_scaled = predict_with_fallback(model, X_new_scaled, y_train_history)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))

# 5. 결과 저장 및 전송
save_prediction(y_pred)
send_to_dashboard(y_pred)

# 6. 성능 모니터링 (실제값 확인 후)
if y_actual_available:
    need_retrain = monitor_model_performance(y_actual, y_pred, threshold_mape=20)
    if need_retrain:
        trigger_retraining()
```

### 4. 배치 예측

```python
# 여러 날짜 예측
dates = pd.date_range('2024-12-01', periods=30, freq='D')
predictions = []

for date in dates:
    X_new = prepare_features_for_date(date)
    X_new_scaled = scaler_X.transform(X_new[feature_cols])
    y_pred_scaled = model.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    predictions.append(y_pred[0])

# DataFrame으로 변환
df_pred = pd.DataFrame(predictions,
                       columns=['호기조_DO', '호기조_NH4_N', '호기조_NO3_N'],
                       index=dates)

print(df_pred)
```

### 5. API 엔드포인트 예시 (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# 모델 로드 (시작 시 1회)
model = joblib.load('model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
feature_cols = joblib.load('feature_cols.pkl')

class InputData(BaseModel):
    pH: float
    수온: float
    DO: float
    MLSS: float
    # ... (모든 약 50개 피처)

class PredictionResponse(BaseModel):
    호기조_DO: float
    호기조_NH4_N: float
    호기조_NO3_N: float
    date: str

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    try:
        # 입력 데이터를 DataFrame으로 변환
        X_new = pd.DataFrame([data.dict()])
        X_new_aligned = X_new[feature_cols]

        # 예측
        X_new_scaled = scaler_X.transform(X_new_aligned)
        y_pred_scaled = model.predict(X_new_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # 응답
        return PredictionResponse(
            호기조_DO=float(y_pred[0][0]),
            호기조_NH4_N=float(y_pred[0][1]),
            호기조_NO3_N=float(y_pred[0][2]),
            date=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행: uvicorn main:app --reload
```

---

## 성능 지표

### 1. 목표 성능

| 지표 | 목표값 | 설명 |
|------|--------|------|
| **R² Score** | ≥ 0.85 | 결정계수 (1에 가까울수록 좋음) |
| **MAPE** | ≤ 15% | 평균 절대 백분율 오차 |
| **RMSE** | DO: ≤ 0.3 mg/L<br>NH4_N: ≤ 0.5 mg/L<br>NO3_N: ≤ 1.0 mg/L | 평균 제곱근 오차 |

### 2. 예상 성능

| 타겟 | Train R² | Test R² | Test MAPE | Test RMSE |
|------|----------|---------|-----------|-----------|
| 호기조 DO | 0.90~0.95 | 0.85~0.90 | 5~8% | 0.2~0.3 mg/L |
| 호기조 NH4_N | 0.90~0.95 | 0.85~0.90 | 10~15% | 0.3~0.5 mg/L |
| 호기조 NO3_N | 0.90~0.95 | 0.85~0.90 | 8~12% | 0.5~1.0 mg/L |

### 3. Time Series Cross-Validation 결과

5-Fold CV 평균 성능 (예상):

```
호기조_DO:    R² = 0.87 ± 0.03
호기조_NH4_N: R² = 0.86 ± 0.04
호기조_NO3_N: R² = 0.85 ± 0.05
```

### 4. Feature Importance Top 10 (예상)

| 순위 | 피처명 | 예상 중요도 | 설명 |
|------|--------|-------------|------|
| 1 | DO(mg/L) | 0.15~0.20 | 현재 DO 상태 |
| 2 | 송풍량(m3/d) | 0.10~0.15 | 공기 공급량 |
| 3 | MLSS(mg/L) | 0.08~0.12 | 미생물 농도 |
| 4 | NH4_N | 0.08~0.12 | 유입 암모니아 |
| 5 | 수온(oC) | 0.05~0.08 | 생물학적 반응 속도 |
| 6 | HRT | 0.05~0.08 | 체류시간 |
| 7 | DO_rolling_mean7 | 0.03~0.05 | DO 트렌드 |
| 8 | season | 0.03~0.05 | 계절성 |
| 9 | MLSS_lag7 | 0.03~0.05 | 과거 MLSS |
| 10 | nitrification_efficiency | 0.03~0.05 | 질산화 효율 |

### 5. 성능 모니터링 기준

#### 정기 재학습 주기
- **기본**: 30일마다 재학습
- **이유**: 계절별 데이터 분포 변화 대응

#### 조기 재학습 기준
- **MAPE > 20%**: 모델 성능 저하 감지 시 즉시 재학습
- **연속 7일 이상 예측 오차 증가**: 트렌드 변화 감지

#### 알림 기준
- **Warning**: MAPE 15~20% (주의 필요)
- **Critical**: MAPE > 20% (재학습 필요)

---

## 참고 문서

- **모델 코드**: [model.ipynb](model.ipynb)
- **리뷰 문서**: [review.md](review.md)
- **데이터셋**: [dataset/호기조.csv](../dataset/호기조.csv)

---

**문의**: 모델 관련 문의사항은 AI팀으로 연락 바랍니다.
**최종 수정일**: 2025-12-17
