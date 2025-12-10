# REQ_ANA001 혐기조 모델 리뷰

**작성일**: 2025-12-10

---

## 📋 모델 개요

- **모델 ID**: REQ_ANA001
- **목적**: 혐기조 BOD/PO4-P/MLSS와 인 방출량, VFA/PAO 지표, BioP 잠재력(%) 동시 예측
- **알고리즘**: MultiOutputRegressor(XGBRegressor, n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0)
- **데이터셋**: `dataset/학습데이터_WWTP_2025_1024.xlsx` (`xy_dataset` 시트, 원본 8,766행 → 기본 전처리 후 8,765행 → lag 적용 후 8,741행)
- **입력 변수**: 원본 30개 + 파생 4개(C_ana_in_PO4P, P_release_conc, P_release_load_gd, BOD_load) + 지표 3개(VFA_index, PAO_index, BioP_potential_pct) + Lag 6/12/24h(주요 공정·타깃 포함) → 총 86개
- **출력 변수**: 혐기_BOD, 혐기_Po4P, 혐기_MLSS, P_release_conc, VFA_index, PAO_index, BioP_potential_pct (7개)
- **데이터 분할**: 시계열 70/15/15 (Train 6,118 / Val 1,311 / Test 1,312), 입력만 MinMaxScaler 적용
- **산출물**: 예측 표와 시각화 3종(`ana_prediction_timeseries_20251210_131813.png`, `ana_prediction_scatter_20251210_131813.png`, `ana_feature_importance_20251210_131813.png`)

---

## 🔴 발생할 수 있는 문제사항

### 1. **혐기_Po4P 예측력 부족**
- **현상**: Test R² 0.142 (MAE 0.281, RMSE 0.365), Val R² 0.561
- **영향**: 인 방출량·BioP 잠재력 해석 시 신뢰도 저하
- **원인**: 입력-타깃 변동성 높음, 도메인 피처 부족, 파라미터 고정

### 2. **일반화/과적합 위험**
- **현상**: Train R² 대부분 0.93~1.0, Val/Test는 0.14~0.99로 갭 존재
- **영향**: 기간·계절 변화 시 성능 하락 가능
- **원인**: 86개 피처(다수 lag) + 단일 시계열 분할 + 조기 종료/강한 정규화 미적용

### 3. **시계열 검증 부재**
- **현상**: 70/15/15 한 번만 분할, TimeSeriesSplit 미적용
- **영향**: 특정 기간에 과적합될 위험, 신뢰구간 부재

### 4. **하이퍼파라미터 튜닝 미흡**
- **현상**: 수동 설정만 사용, 탐색 없이 n_estimators=500 고정
- **영향**: 훈련 시간 대비 효율·성능 최적화 미흡

### 5. **Lag 기반 Cold Start**
- **현상**: 6/12/24h lag으로 초기 24시간 예측 불가, 실시간 결측 시 대체 로직 없음
- **영향**: 배포 초기·센서 결측 시 서비스 중단 위험

### 6. **결측/이상치 처리 단순**
- **현상**: 선형 보간 후 남은 결측만 drop, 이상치 클리핑 부재
- **영향**: 극단값이 BOD_load, P_release_load_gd 등 부하 피처에 과도한 영향

### 7. **파이프라인 저장 미비**
- **현상**: 스케일러·모델 저장/버전 관리 없음
- **영향**: 재현성 및 배포 시 동일 스케일 재사용 불가

### 8. **지표 다양성 부족**
- **현상**: MAE/RMSE/R²만 산출, 상대오차/분포 기반 지표 부재
- **영향**: 저농도 구간(Po4P) 민감도 파악 어려움

---

## ✅ 개선방안 제안

### 1. **Po4P/PAO 예측력 보강**

**방법**: 타깃 가중치 부여 + Po4P 전용 파라미터 미세조정

```python
# Po4P 저농도 구간 가중치
weights = np.where(
    y_train[:, target_cols.index("혐기_Po4P")] < 0.5,
    1.5,
    1.0
)

xgb_base = XGBRegressor(
    n_estimators=600, learning_rate=0.03, max_depth=5,
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=0.2, reg_lambda=1.2,
    objective="reg:squarederror", random_state=42
)

model = MultiOutputRegressor(xgb_base)
model.fit(X_train_scaled, y_train, sample_weight=weights)
```

**기대 효과**: 저농도 구간 포착 → Po4P/PAO R² 0.6 이상 목표

---

### 2. **조기 종료 + 정규화 강화**

**방법**: 검증 세트 분리 후 early_stopping 적용, min_child_weight/gamma 조정

```python
xgb_base = XGBRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=4,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, gamma=0.1,
    objective="reg:squarederror", random_state=42
)

for i, target in enumerate(target_cols):
    xgb_base.fit(
        X_train_scaled, y_train[:, i],
        eval_set=[(X_val_scaled, y_val[:, i])],
        eval_metric=["rmse", "mae"],
        early_stopping_rounds=50,
        verbose=False
    )
```

**기대 효과**: 과적합 완화, Val/Test 갭 축소

---

### 3. **Time Series Cross-Validation 도입**

**방법**: TimeSeriesSplit 기반 성능 분포 확인

```python
tscv = TimeSeriesSplit(n_splits=5)
r2_hist = {t: [] for t in target_cols}

for tr_idx, va_idx in tscv.split(X):
    model.fit(X[tr_idx], y[tr_idx])
    pred = model.predict(X[va_idx])
    for i, t in enumerate(target_cols):
        r2_hist[t].append(r2_score(y[va_idx, i], pred[:, i]))

for t, scores in r2_hist.items():
    print(f"{t}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**기대 효과**: 계절성/기간별 안정성 확인, 배포 위험도 감소

---

### 4. **계절·지연 피처 확장**

**방법**: 달/요일/시간대 + 48/72h lag, rolling 통계 추가

```python
df["month"] = df["Time"].dt.month
df["hour"] = df["Time"].dt.hour
df["dayofweek"] = df["Time"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

extra_lags = [48, 72]
for feat in ["유입유량", "유입BOD", "유입TP", "Q_RAS (반송)", "총인처리 약품주입률"]:
    for lag in extra_lags:
        df_lag[f"{feat}_lag{lag}"] = df[feat].shift(lag)
    df_lag[f"{feat}_roll24_mean"] = df[feat].rolling(24).mean()
```

**기대 효과**: 중·장주기 패턴 반영 → BOD/Po4P 안정성 향상

---

### 5. **하이퍼파라미터 탐색**

**방법**: RandomizedSearchCV + TimeSeriesSplit

```python
param_dist = {
    "estimator__max_depth": [3, 4, 5, 6],
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "estimator__n_estimators": [300, 500, 700],
    "estimator__subsample": [0.7, 0.85, 1.0],
    "estimator__colsample_bytree": [0.7, 0.85, 1.0],
    "estimator__min_child_weight": [1, 3, 5],
    "estimator__gamma": [0, 0.05, 0.1]
}

search = RandomizedSearchCV(
    MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", random_state=42)),
    param_dist, n_iter=30, cv=TimeSeriesSplit(n_splits=3),
    scoring="r2", n_jobs=-1, random_state=42
)
search.fit(X_train_scaled, y_train)
print(search.best_params_)
```

**기대 효과**: 튜닝 없이 고정된 500트리 대비 효율·성능 개선

---

### 6. **결측/이상치 견고화**

**방법**: IQR 클리핑 + 최대 3시간 ffill/bfill

```python
def robust_clean(df, num_cols):
    q1, q3 = df[num_cols].quantile(0.01), df[num_cols].quantile(0.99)
    df[num_cols] = df[num_cols].clip(q1, q3)
    df[num_cols] = df[num_cols].interpolate("linear").ffill().bfill(limit=3)
    return df

df = robust_clean(df, ["유입유량", "유입BOD", "유입TP", "Q_RAS (반송)", "BOD_load", "P_release_load_gd"])
```

**기대 효과**: 급변 구간 노이즈 완화 → 오차 폭 감소

---

### 7. **파이프라인/아티팩트 저장**

**방법**: 입력 스케일러·모델 동시 저장 및 버전 태깅

```python
from joblib import dump

dump(x_scaler, f"artifacts/x_scaler_{timestamp}.joblib")
dump(model, f"artifacts/xgb_multi_{timestamp}.joblib")
```

**기대 효과**: 재현성 확보, 배포/재학습 자동화 용이

---

### 8. **Cold Start 및 모니터링**

**방법**: Lag 결측 시 평균 기반 대체, MAPE 모니터링 후 재학습 트리거

```python
def predict_with_fallback(model, X_row, history, lag_cols):
    if X_row[lag_cols].isnull().any():
        return history.tail(24).mean().values  # 최근 24시간 평균
    return model.predict(X_row)

def needs_retrain(y_true, y_pred, threshold=20):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mape > threshold
```

**기대 효과**: 초기 24시간 및 센서 결측 상황에서도 서비스 지속

---

## 📊 개선 우선순위

### 🔴 긴급 (High Priority)
1. Po4P/PAO 성능 보강 및 과적합 완화 (가중치 + 조기 종료)
2. TimeSeriesSplit 기반 검증 도입
3. 결측/Cold Start 대응 로직 추가

### 🟡 중요 (Medium Priority)
4. 하이퍼파라미터 탐색으로 파라미터 최적화
5. 계절·지연 피처 확장 및 도메인 피처 보강
6. 파이프라인/아티팩트 저장 및 버전 관리

### 🟢 권장 (Low Priority)
7. 지표 다양화(MAPE/sMAPE, 분포 기반) 및 모니터링 대시보드
8. 추가 시각화·설명력(Feature Importance, SHAP) 업데이트

---

## 🎯 기대 효과

- Po4P/PAO R² 0.6 이상, BOD/MLSS/VFA/BioP R² 0.9 이상 유지
- Val/Test 간 격차 축소 및 기간별 성능 분산 확인
- 초기 24시간·센서 결측 시 예측 지속, 재현 가능한 배포 아티팩트 확보

---

## 📝 결론

현재 파이프라인은 **데이터 로드 → 파생/지표 생성 → 6/12/24h lag → 70/15/15 분할 → MinMax 스케일 → XGB 다중회귀 학습** 흐름을 완료했고, 다수 타깃에서 양호한 성능을 보였다. 그러나 **Po4P/PAO 예측력**과 **검증/운영 전략 부재**가 주요 리스크이다. 상위 우선순위 개선(가중치+조기 종료+시계열 CV+Cold Start 로직)을 적용하면 배포 신뢰도가 크게 향상될 것으로 판단된다.

---

## ✅ 적용 현황 (req_anaerobic_model.ipynb 기준)

### 적용 완료
- 데이터 로드/기본 전처리 (Cell #1-2): 수치 변환, 선형 보간, 결측 제거
- 파생 컬럼/지표 생성 (Cell #3-4): C_ana_in_PO4P, P_release_conc/load, BOD_load, VFA/PAO/BioP 지표
- Lag 피처 6/12/24h 생성 (Cell #5): 주요 공정·타깃 대상, 최종 8,741행/86피처
- 시계열 분할 및 스케일링 (Cell #6-7): Train 6,118 / Val 1,311 / Test 1,312, 입력 MinMaxScaler
- 모델 학습 (Cell #8): MultiOutputRegressor(XGBRegressor: n_estimators=500, depth=4, lr=0.05, subsample/colsample=0.9, reg_alpha=0.1, reg_lambda=1.0)
- 성능 평가 및 시각화 (Cell #9-10): Train/Val/Test MAE·RMSE·R² 계산, 예측 시계열·산점도·Feature Importance 저장

### 미적용/추가 필요
- TimeSeriesSplit 기반 교차검증, 조기 종료, 가중치 학습
- 하이퍼파라미터 탐색(Randomized/Optuna 등)
- 결측/이상치 견고화, Cold Start fallback, 파이프라인/모델 저장 및 모니터링

---

## 📈 셋별 성능 비교

| target | Train R² | Val R² | Test R² |
| --- | ---: | ---: | ---: |
| 혐기_BOD | 0.9334 | 0.8513 | 0.8395 |
| 혐기_Po4P | 0.8681 | 0.5611 | 0.1421 |
| 혐기_MLSS | 0.99997 | 0.99988 | 0.99976 |
| P_release_conc | 0.99990 | 0.99859 | 0.99546 |
| VFA_index | 0.99927 | 0.99791 | 0.99637 |
| PAO_index | 0.99989 | 0.94613 | 0.76160 |
| BioP_potential_pct | 0.99905 | 0.98723 | 0.94865 |

추가 요약: Test MAE/RMSE — BOD 0.903/1.170, Po4P 0.281/0.365, MLSS 3.095/4.267, P_release 0.006/0.023, VFA 0.003/0.006, PAO 0.038/0.040, BioP 1.241/1.529

---

## 🔄 후속 작업 권장사항

1. 상위 우선순위 개선 3건(Po4P 가중치/조기 종료/TS CV) 적용 후 재학습 및 동일 스플릿 성능 재평가
2. RandomizedSearchCV로 파라미터 탐색 → best 모델/스케일러 아티팩트 저장
3. Cold Start·결측 대응 로직 적용 후 실시간 시나리오 테스트(초기 24h, 센서 드롭)
4. Po4P/PAO 에러 분석(잔차 플롯, 중요도 상위 피처 점검) 및 추가 피처 후보 선정
5. 리포트/대시보드 업데이트: 최신 예측 그래프, 지표(MAPE/sMAPE) 포함

---

## 📎 참고 파일

- 모델 노트북: `03 혐기조/req_anaerobic_model.ipynb`
- 시각화: `03 혐기조/ana_prediction_timeseries_20251210_131813.png`, `03 혐기조/ana_prediction_scatter_20251210_131813.png`, `03 혐기조/ana_feature_importance_20251210_131813.png`
- 데이터셋: `dataset/학습데이터_WWTP_2025_1024.xlsx` (`xy_dataset` 시트)
- 리포트: `03 혐기조/anaerobic_full_report.md`

---

## 📚 추가 문서

- `03 혐기조/anaerobic_full_report.md`: 현재 모델 단계별 그래프/설명 요약. 향후 개선 적용 시 함께 업데이트 권장.

---

## 2025-12-10 작업 메모

- nbclient로 전체 재실행, Train/Val/Test 성능 및 시각화 최신화. Po4P 성능은 여전히 낮아 후속 조치 필요.
- PowerShell(cp949) 환경을 고려해 로그/파일명은 ASCII로 유지(`[1/10]` 형식 사용, 완료 메시지 `[OK]`).
