# MFR-008: 운전 및 운영 제어인자 도출 모델

- **과업지시서**: MFR-008 — 운전 및 운영 제어인자 도출 모델. 개별 제어인자(송풍량, 외부탄소원, 응집제)에 대한 최적값 도출
- **데이터정의서 모델**: 최적제어 도출 모델
- **기능명세서**: FUN-003-0100 (송풍량 최적화), FUN-003-0200 (외부탄소원 최적화), FUN-003-0300 (응집제 최적화)
- **Spec 모듈**: `006-control-optimization`

> **참고**: 기능명세서에서 MFR-008(송풍량), MFR-009(외부탄소원), MFR-010(응집제)으로 분리 참조되는 기능들을 과업지시서 MFR-008 하에 통합 정리합니다.

---

## 설계 기준 참조

> 제어인자 도출의 기준이 되는 장비 사양 및 운전 범위입니다.

**송풍기 사양**:

| 항목 | 설계값 | 단위 |
|------|--------|------|
| 송풍기 대수 | 6 | 대 |
| 단위 용량 | 300 | m³/min |
| 총 용량 | 1,800 | m³/min |
| 산기장치 형식 | 미세기포 (fine bubble) | - |
| 산기장치 수심 | 5.5 | m |
| 송풍기 효율 | 72 | % |
| 모터 효율 | 93 | % |

**외부탄소원 사양**:

| 항목 | 설계값 | 단위 |
|------|--------|------|
| 탄소원 종류 | 메탄올 | - |
| 최대 투입량 | 20,000 | kg/d |
| 투입 위치 | 무산소조 | - |
| 제어 방식 | C/N 피드포워드 + NO₃ 피드백 | - |
| NO₃ 목표 | 2.0 | mg/L |
| C/N 투입 기준 | < 6.0 | - |

**응집제 사양**:

| 항목 | 설계값 | 단위 |
|------|--------|------|
| 응집제 종류 | PAC (Poly Aluminum Chloride) | - |
| 기본 투입량 | 30 | mg/L |
| 최대 투입량 | 100 | mg/L |
| 인 제거율 | 85 | % |
| 동적 제어 | ON | - |

**DO 제어 범위**:

| 항목 | 설계값 | 단위 |
|------|--------|------|
| DO 기본 설정 | 2.0 | mg/L |
| DO 범위 | 1.5 ~ 3.5 | mg/L |
| NH₄ 기반 동적제어 | ON | - |
| NH₄ 목표 | 1.0 | mg/L |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

---

## 1. 송풍량 최적화

### `POST /optimization/aeration`

AI가 추천하는 최적 송풍량과 DO 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0100 (송풍량 최적화)

**Request Body**:

```json
{
  "current_state": {
    "influent": { "Q": 10000, "BOD": 150 },
    "aerobic": { "DO": 2.1, "NH4": 1.2, "MLSS": 3500 },
    "current_airflow": 62.5
  },
  "target": { "DO": 2.0 },
  "cost_info": {
    "electricity_unit_price": 120
  }
}
```

> **데이터 출처**: `dataset/a2o/results_ml_measurable.csv`에서 제공되는 데이터 사용
> - **CSV 제공**: 유입_Q, 유입_BOD, 호기_DO, 호기_S_NH, 호기_MLSS, 송풍_Q_air
> - **설정값**: 호기_DO_setpoint (목표값은 설정 파일/DB 관리)
> - **비용 정보**: 전력 단가 (선택 사항, 설정 파일/DB 관리)

| 필드 | 타입 | 필수 | 단위 | 설명 | CSV 컬럼 |
|------|------|------|------|------|---------|
| `current_state.influent.Q` | float | Y | m³/day | 유입유량 | 유입_Q |
| `current_state.influent.BOD` | float | Y | mg/L | 유입 BOD | 유입_BOD |
| `current_state.aerobic.DO` | float | Y | mg/L | 현재 호기조 DO | 호기_DO |
| `current_state.aerobic.NH4` | float | Y | mg/L | 현재 NH₄-N | 호기_S_NH |
| `current_state.aerobic.MLSS` | float | Y | mg/L | 호기조 MLSS | 호기_MLSS |
| `current_state.current_airflow` | float | Y | m³/min | 현재 총 송풍량 | 송풍_Q_air |
| `target.DO` | float | Y | mg/L | DO 목표값 | 호기_DO_setpoint |
| `cost_info.electricity_unit_price` | float | N | 원/kWh | 전력 단가 (선택) | - |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "recommended_airflow": {
      "total": { "value": 60.0, "unit": "m³/min" },
      "change": { "value": -2.5, "percent": -4.0 }
    },
    "nh4n_do_status": {
      "current_NH4N": 1.2,
      "target_DO": 2.0,
      "control_status": "정상"
    },
    "cost_benefit": {
      "daily": { "DO_achievement_rate": 95.0, "power_usage": 2880, "power_usage_unit": "kWh" },
      "weekly": { "DO_achievement_rate": 93.5, "power_usage": 20160 },
      "monthly": { "DO_achievement_rate": 94.2, "power_usage": 86400 },
      "savings": {
        "monthly": { "value": 54, "unit": "만원" },
        "annual": { "value": 648, "unit": "만원" }
      }
    }
  },
  "metadata": {
    "fun_id": "FUN-003-0100",
    "optimization_cycle": "1h",
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **cost_benefit 계산 근거**:
>
> ```
> ## 전력 사용량 (power_usage)
> P(kW) = (Q_air_m³s × ΔP) / (η_blower × η_motor)
> - ΔP = ρ × g × h = 1000 × 9.81 × 5.5 = 53,955 Pa
> - η_blower = 0.72, η_motor = 0.93
> - Q_air_m³s = Q_air_m³min / 60
> - power_usage_daily(kWh) = P(kW) × 24
>
> ## 절감액 (savings)
> - daily_saving_kWh = current_power - recommended_power
> - monthly_saving(만원) = daily_saving_kWh × 30 × electricity_unit_price / 10000
> - annual_saving(만원) = monthly_saving × 12
> ```

---

## 2. 외부탄소원 최적화

### `POST /optimization/carbon`

AI가 추천하는 최적 외부탄소원 투입량과 TN 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0200 (외부탄소원 최적화)

**Request Body**:

```json
{
  "current_state": {
    "influent": { "Q": 10000, "TN": 35.0, "BOD": 150 },
    "anoxic": { "S_NO": 2.1, "S_NH": 12.0, "MLSS": 2900 },
    "effluent": { "TN": 14.1 },
    "current_carbon_dose": 100
  },
  "target": {
    "TN_discharge_limit": 20.0,
    "TN_operation_target": 7.0
  },
  "cost_info": {
    "carbon_unit_price": 850
  }
}
```

> **데이터 출처**: `dataset/a2o/results_ml_measurable.csv`에서 제공되는 데이터 사용
> - **CSV 제공**: 유입_Q, 유입_TN, 유입_BOD, 무산소_S_NO, 무산소_S_NH, 무산소_MLSS, 방류_TN, 운전_carbon_kg_d
> - **설정값**: TN 운전 목표 (설정 파일/DB 관리)
> - **법적 기준**: TN 방류기준 ≤20 mg/L (고정값)
> - **비용 정보**: 탄소원 단가 (선택 사항, 설정 파일/DB 관리)

| 필드 | 타입 | 필수 | 단위 | 설명 | CSV 컬럼 |
|------|------|------|------|------|---------|
| `current_state.influent.Q` | float | Y | m³/day | 유입유량 | 유입_Q |
| `current_state.influent.TN` | float | Y | mg/L | 유입 TN | 유입_TN |
| `current_state.influent.BOD` | float | Y | mg/L | 유입 BOD | 유입_BOD |
| `current_state.anoxic.S_NO` | float | Y | mg/L | 무산소조 NO₃-N | 무산소_S_NO |
| `current_state.anoxic.S_NH` | float | Y | mg/L | 무산소조 NH₄-N | 무산소_S_NH |
| `current_state.anoxic.MLSS` | float | Y | mg/L | 무산소조 MLSS | 무산소_MLSS |
| `current_state.effluent.TN` | float | Y | mg/L | 방류 TN | 방류_TN |
| `current_state.current_carbon_dose` | float | Y | kg/day | 현재 탄소원 투입량 | 운전_carbon_kg_d |
| `target.TN_discharge_limit` | float | Y | mg/L | TN 방류기준 (≤20) | - |
| `target.TN_operation_target` | float | Y | mg/L | TN 운전목표 | - |
| `cost_info.carbon_unit_price` | float | N | 원/kg | 탄소원 단가 (선택) | - |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "recommended_dose": {
      "value": 110, "unit": "kg/day",
      "change": { "value": 10, "percent": 10.0 }
    },
    "tn_status": {
      "current_TN": 14.1,
      "target_TN": 7.0,
      "control_status": "목표 미달 — 투입량 증가 권고"
    },
    "cost_benefit": {
      "daily": { "dose_volume": 110, "TN_removal_rate": 65.7 },
      "weekly": { "dose_volume": 770, "TN_removal_rate": 64.2 },
      "monthly": { "dose_volume": 3300, "TN_removal_rate": 65.0 },
      "improvement_rate": 8.5,
      "cost_change_percent": 10.0,
      "investment_grade": "우수"
    }
  },
  "metadata": {
    "fun_id": "FUN-003-0200",
    "optimization_cycle": "1h",
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **cost_benefit 계산 근거**:
>
> ```
> ## 투입량 (dose_volume)
> - daily = recommended_dose (kg/day)
> - weekly = daily × 7, monthly = daily × 30
>
> ## TN 제거율 (TN_removal_rate)
> - TN_removal_rate(%) = (influent_TN - effluent_TN) / influent_TN × 100
>
> ## 비용 변동률 (cost_change_percent)
> - cost_change_percent(%) = (recommended - current) / current × 100
>
> ## 투자등급 (investment_grade)
> - cost↓ and TN < target → "우수"
> - TN < target → "양호"
> - TN < discharge_limit → "보통"
> - else → "미흡"
> ```

---

## 3. 응집제 최적화

### `POST /optimization/coagulant`

AI가 추천하는 최적 응집제 투입량과 TP 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0300 (응집제 최적화)

**Request Body**:

```json
{
  "current_state": {
    "influent": { "Q": 10000, "TP": 5.0, "BOD": 150 },
    "anaerobic": { "S_PO4": 8.5, "MLSS": 2800 },
    "aerobic": { "S_PO4": 0.5, "MLSS": 3500 },
    "effluent": { "TP": 0.18 },
    "current_coagulant_dose": 50
  },
  "target": {
    "TP_discharge_limit": 0.2,
    "TP_operation_target": 0.3
  },
  "cost_info": {
    "coagulant_unit_price": 1200
  }
}
```

> **데이터 출처**: `dataset/a2o/results_ml_measurable.csv`에서 제공되는 데이터 사용
> - **CSV 제공**: 유입_Q, 유입_TP, 유입_BOD, 혐기_S_PO4, 혐기_MLSS, 호기_S_PO4, 호기_MLSS, 방류_TP, 총인처리_coag_kg_d
> - **설정값**: TP 운전 목표 (설정 파일/DB 관리)
> - **법적 기준**: TP 방류기준 ≤0.2 mg/L (I지역 기준, 고정값)
> - **비용 정보**: 응집제 단가 (선택 사항, 설정 파일/DB 관리)

| 필드 | 타입 | 필수 | 단위 | 설명 | CSV 컬럼 |
|------|------|------|------|------|---------|
| `current_state.influent.Q` | float | Y | m³/day | 유입유량 | 유입_Q |
| `current_state.influent.TP` | float | Y | mg/L | 유입 TP | 유입_TP |
| `current_state.influent.BOD` | float | Y | mg/L | 유입 BOD | 유입_BOD |
| `current_state.anaerobic.S_PO4` | float | Y | mg/L | 혐기조 PO₄-P | 혐기_S_PO4 |
| `current_state.anaerobic.MLSS` | float | Y | mg/L | 혐기조 MLSS | 혐기_MLSS |
| `current_state.aerobic.S_PO4` | float | Y | mg/L | 호기조 PO₄-P | 호기_S_PO4 |
| `current_state.aerobic.MLSS` | float | Y | mg/L | 호기조 MLSS | 호기_MLSS |
| `current_state.effluent.TP` | float | Y | mg/L | 방류 TP | 방류_TP |
| `current_state.current_coagulant_dose` | float | Y | kg/day | 현재 응집제 투입량 | 총인처리_coag_kg_d |
| `target.TP_discharge_limit` | float | Y | mg/L | TP 방류기준 (≤0.2) | - |
| `target.TP_operation_target` | float | Y | mg/L | TP 운전목표 | - |
| `cost_info.coagulant_unit_price` | float | N | 원/kg | 응집제 단가 (선택) | - |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "recommended_dose": {
      "value": 48, "unit": "kg/day",
      "change": { "value": -2, "percent": -4.0 }
    },
    "tp_status": {
      "current_TP": 0.18,
      "target_TP": 0.3,
      "control_status": "목표 충족 — 약품 절감 가능"
    },
    "cost_benefit": {
      "daily": { "dose_volume": 48, "TP_removal_rate": 96.4 },
      "weekly": { "dose_volume": 336, "TP_removal_rate": 96.0 },
      "monthly": { "dose_volume": 1440, "TP_removal_rate": 95.8 },
      "savings": {
        "monthly": { "value": 7.2, "unit": "만원" },
        "annual": { "value": 86.4, "unit": "만원" }
      }
    }
  },
  "metadata": {
    "fun_id": "FUN-003-0300",
    "optimization_cycle": "1h",
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **cost_benefit 계산 근거**:
>
> ```
> ## 투입량 (dose_volume)
> - daily = recommended_dose (kg/day)
> - weekly = daily × 7, monthly = daily × 30
>
> ## TP 제거율 (TP_removal_rate)
> - TP_removal_rate(%) = (influent_TP - effluent_TP) / influent_TP × 100
>
> ## 절감액 (savings)
> - daily_saving(원) = (current - recommended) × coagulant_unit_price
> - monthly_saving(만원) = daily_saving × 30 / 10000
> - annual_saving(만원) = monthly_saving × 12
> ```

---

## 참고 문서

### 구현 가이드

본 API의 상세 구현 방법은 다음 문서를 참조하세요:

- **구현 가이드**: [docs/specs/MFR-008-control-derivation.md](../specs/MFR-008-control-derivation.md)
  - ML 모델 학습 방법 (XGBoost, LSTM)
  - 최적화 알고리즘 (scipy.optimize)
  - 비용 편익 계산
  - 모델 관리 및 재학습 전략

### 관련 문서

- **이슈 정리**: [docs/ISSUE.md](../ISSUE.md) - MFR-008 섹션
- **CSV 데이터**: `dataset/a2o/results_ml_measurable.csv`

---

**작성자**: Claude Sonnet 4.5
**최종 수정**: 2026-02-26
