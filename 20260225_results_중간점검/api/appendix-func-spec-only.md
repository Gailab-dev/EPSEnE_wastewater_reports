# 부록: 기능명세서 전용 API (과업지시서 MFR 해당 없음)

아래 API들은 기능명세서(A2O-FUNC-SPEC-v5.1)에 정의되어 있으나, 과업지시서 MFR-003~MFR-008에 직접 매핑되지 않는 기능입니다.

**총 9개 API** (구현 대상 외):

| # | Method | Endpoint | FUN ID | 설명 |
|---|--------|----------|--------|------|
| 1 | POST | `/analytics/statistics` | FUN-005-0500 | 통계 분석 |
| 2 | POST | `/analytics/correlation` | FUN-005-0400 | 상관관계 분석 |
| 3 | GET | `/dashboard/optimization-summary` | FUN-000-0100 | 대시보드 최적화 요약 |
| 4 | GET | `/bioreactor/ai-auto-status` | FUN-001-0400 | AI 자동 제어 상태 |
| 5 | POST | `/analytics/influent-analysis` | FUN-002-0100 | 유입수 종합 분석 |
| 6 | POST | `/analytics/bioreactor-analysis` | FUN-002-0200 | 생물반응조 분석·진단 |
| 7 | GET | `/data/raw` | FUN-005-0100 | 원천데이터 조회 |
| 8 | GET | `/data/function/{function_name}` | FUN-005-0200 | 기능별 데이터 조회 |
| 9 | POST | `/data/multi-trend` | FUN-005-0300 | 다중 추이 분석 |

---

## 1. 통계 분석

### `POST /analytics/statistics`

계측요소의 기술통계량을 산출합니다.

> **기능명세서**: FUN-005-0500 (계측요소별 통계 분석)
> **과업지시서 매핑**: 해당 없음

**Request Body**:

```json
{
  "parameter": "aerobic_DO",
  "period": { "start": "2026-01-01", "end": "2026-01-31" },
  "interval": "1h"
}
```

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "parameter": "aerobic_DO",
    "statistics": {
      "count": 744,
      "mean": 2.15,
      "median": 2.10,
      "std": 0.35,
      "min": 1.20,
      "max": 3.40,
      "cv": 16.28,
      "Q1": 1.90,
      "Q2": 2.10,
      "Q3": 2.40,
      "IQR": 0.50,
      "skewness": 0.23,
      "kurtosis": -0.15
    },
    "timeseries": [],
    "boxplot_data": {},
    "histogram_data": {}
  },
  "metadata": {
    "fun_id": "FUN-005-0500",
    "mfr_mapping": "과업지시서 MFR 해당 없음"
  }
}
```

---

## 2. 상관관계 분석

### `POST /analytics/correlation`

공정별 계측요소 간 상관관계를 분석합니다 (히트맵, 산점도, 군집 레이더).

> **기능명세서**: FUN-005-0400 (공정별 계측요소 분석)
> **과업지시서 매핑**: 해당 없음

**Request Body**:

```json
{
  "process": "aerobic",
  "parameters": ["MLSS", "DO", "S_NH", "S_NO", "S_PO4", "pH", "SVI"],
  "period": { "start": "2026-01-01", "end": "2026-01-31" }
}
```

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "correlation_matrix": {
      "MLSS": { "MLSS": 1.0, "DO": -0.32, "S_NH": -0.45, "SVI": 0.28 },
      "DO": { "MLSS": -0.32, "DO": 1.0, "S_NH": -0.68, "SVI": -0.15 }
    },
    "regression": {
      "x": "DO", "y": "S_NH",
      "slope": -3.25, "intercept": 8.12, "r_squared": 0.462
    },
    "cluster_radar": [
      { "cluster_id": 0, "label": "정상 운전", "profile": { "MLSS": 0.65, "DO": 0.72, "SVI": 0.55 } },
      { "cluster_id": 1, "label": "고부하 운전", "profile": { "MLSS": 0.85, "DO": 0.45, "SVI": 0.78 } }
    ]
  },
  "metadata": {
    "fun_id": "FUN-005-0400",
    "mfr_mapping": "과업지시서 MFR 해당 없음"
  }
}
```

---

## 3. 대시보드 최적화 권고 조회

### `GET /dashboard/optimization-summary`

> **기능명세서**: FUN-000-0100 (통합 상황판) — AI API 시트 "송풍량/외부탄소원/응집제 최적화 권고"
> **과업지시서 매핑**: 해당 없음 (MFR-008 하위 API의 요약 뷰)

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "aeration": {
      "current_airflow": 3750,
      "recommended_airflow": 3600,
      "savings_monthly": 54,
      "unit": "만원"
    },
    "carbon": {
      "current_dose": 100,
      "recommended_dose": 110,
      "improvement_rate": 8.5,
      "unit": "%"
    },
    "coagulant": {
      "current_dose": 50,
      "recommended_dose": 48,
      "savings_monthly": 7.2,
      "unit": "만원"
    }
  }
}
```

---

## 4. 생물반응조 AI 자동 제어 상태 조회

### `GET /bioreactor/ai-auto-status`

> **기능명세서**: FUN-001-0400 (생물반응조 감시) — AI API 시트 "현재 제어상태 중 AI 자동 자료"
> **과업지시서 매핑**: 해당 없음

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "ai_auto_mode": true,
    "controls": {
      "aeration": { "mode": "AI_AUTO", "setpoint": 3600, "unit": "m³/hr" },
      "carbon": { "mode": "MANUAL", "setpoint": 100, "unit": "L/day" },
      "coagulant": { "mode": "AI_AUTO", "setpoint": 48, "unit": "L/day" },
      "IR": { "mode": "AI_AUTO", "setpoint": 200, "unit": "%" },
      "RAS": { "mode": "MANUAL", "setpoint": 50, "unit": "%" },
      "WAS": { "mode": "AI_AUTO", "setpoint": 220, "unit": "m³/day" }
    },
    "last_updated": "2026-02-19T10:00:00+09:00"
  }
}
```

---

## 5. 유입수 종합 분석 (군집 판정 + AI 진단)

### `POST /analytics/influent-analysis`

유입수의 부하 상태, 수질 지표, 변동성을 종합 분석하고 AI가 군집 판정 및 진단을 수행합니다.

> **기능명세서**: FUN-002-0100 (유입수 분석)
> **과업지시서 매핑**: 해당 없음

**Request Body**:

```json
{
  "current": {
    "influent_flow": 115000,
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "SS": 120.0,
    "temperature": 18.5
  },
  "design_capacity": {
    "flow": 115000,
    "BOD_load": 23000,
    "TN_load": 4600,
    "TP_load": 460
  },
  "analysis_period_days": 60
}
```

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "load_rates": {
      "flow": { "value": 66.7, "unit": "%", "status": "normal" },
      "BOD": { "value": 88.9, "unit": "%", "status": "normal" },
      "TN": { "value": 100.0, "unit": "%", "status": "warning" },
      "TP": { "value": 100.0, "unit": "%", "status": "warning" }
    },
    "quality_indicators": {
      "BOD_COD_ratio": { "value": 0.60, "status": "양호", "description": "생분해성 양호" },
      "CN_ratio": { "value": 4.29, "status": "주의", "description": "탈질 탄소원 부족 가능" }
    },
    "clustering": {
      "cluster_name": "정상유입군",
      "confidence": 87.3,
      "scatter_coordinates": { "PC1": 1.23, "PC2": -0.45 }
    },
    "diagnosis": {
      "load_status": { "label": "부하율 안정", "description": "설계 대비 67-100% 운영" },
      "recommendations": ["외부탄소원 주입량 증가 검토 (C/N비 개선)"]
    }
  },
  "metadata": {
    "fun_id": "FUN-002-0100",
    "mfr_mapping": "과업지시서 MFR 해당 없음"
  }
}
```

---

## 6. 생물반응조 종합 분석·진단

### `POST /analytics/bioreactor-analysis`

생물반응조의 핵심 지표, 공정별 효율, PCA/군집분석, 슬러지 관리 AI 진단을 수행합니다.

> **기능명세서**: FUN-002-0200 (생물반응조 분석·진단)
> **과업지시서 매핑**: 해당 없음

**Request Body**:

```json
{
  "measurements": {
    "anaerobic": { "MLSS": 2800, "DO": 0.2, "ORP": -180, "S_PO4": 8.5 },
    "anoxic": { "MLSS": 2900, "DO": 0.3, "ORP": -50, "S_NO": 2.1, "S_NH": 15.0 },
    "aerobic": { "MLSS": 3500, "DO": 2.1, "SVI": 120 }
  },
  "operation": {
    "airflow": 5000,
    "RAS_rate": 50,
    "IR_rate": 200,
    "WAS_volume": 200
  },
  "influent": { "flow": 115000, "BOD": 150, "TN": 35, "TP": 5 },
  "reactor_volumes": { "anaerobic": 8000, "anoxic": 16500, "aerobic": 36806 }
}
```

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "core_indicators": {
      "SRT": { "value": 17.5, "unit": "day", "status": "normal" },
      "FM_ratio": { "value": 0.107, "unit": "kgBOD/kgMLSS·d", "status": "normal" },
      "MLSS": { "value": 3500, "unit": "mg/L", "status": "normal" }
    },
    "sludge_management": {
      "recommended_WAS": { "value": 220, "unit": "m³/day" },
      "recommended_IR": { "value": 210, "unit": "%" }
    },
    "clustering": {
      "cluster_name": "정상운전군",
      "confidence": 91.2,
      "recommendations": {
        "DO_target": { "value": 2.0, "unit": "mg/L" }
      }
    }
  },
  "metadata": {
    "fun_id": "FUN-002-0200",
    "mfr_mapping": "과업지시서 MFR 해당 없음"
  }
}
```

---

## 7. 원천데이터 조회

### `GET /data/raw`

> **기능명세서**: FUN-005-0100 (원천데이터 조회)
> **과업지시서 매핑**: 해당 없음

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `start_date` | string | Y | 시작일 |
| `end_date` | string | Y | 종료일 |
| `interval` | string | Y | `1min` / `10min` / `1h` / `1d` |
| `process` | string | N | 공정 필터 |
| `data_type` | string | N | `scada` / `lab` / `all` |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "scada": {
      "count": 1440,
      "records": [
        {
          "timestamp": "2026-02-19T00:00:00+09:00",
          "influent": { "Q": 420.5, "T": 17.8, "pH": 7.1 },
          "aerobic": { "MLSS": 3480, "DO": 2.05, "pH": 7.2 }
        }
      ]
    },
    "lab": {
      "count": 1,
      "records": [
        {
          "timestamp": "2026-02-19T09:00:00+09:00",
          "influent": { "BOD": 148, "COD": 245, "TN": 34, "TP": 4.8, "SS": 118 },
          "effluent": { "BOD": 4.8, "COD": 15.2, "TN": 14.1, "TP": 0.18, "SS": 3.5 }
        }
      ]
    }
  },
  "metadata": {
    "fun_id": "FUN-005-0100",
    "max_query_period": "1 year"
  }
}
```

---

## 8. 기능별 데이터 조회

### `GET /data/function/{function_name}`

> **기능명세서**: FUN-005-0200 (기능별 데이터 조회)
> **과업지시서 매핑**: 해당 없음

**Path Parameters**:

| function_name | 설명 | 필수 데이터 항목 |
|--------------|------|----------------|
| `aeration` | 송풍량 최적화 | 호기_DO, MLSS, pH, 송풍_Q, 유입_Q, 전력 |
| `carbon` | 외부탄소원 최적화 | 무산소_ORP, 탄소원_주입량, 무산소_DO |
| `coagulant` | 응집제 최적화 | 응집제_주입량, 방류_pH |
| `influent-analysis` | 유입수 분석 | 유입_Q, T, pH, BOD, COD, TN, TP, SS |
| `bioreactor` | 생물반응조 진단 | 혐기/무산소/호기_MLSS, DO, ORP |
| `effluent-prediction` | 방류수 예측 | 유입_Q, 호기_DO, MLSS, 송풍_Q |

---

## 9. 다중 추이 분석

### `POST /data/multi-trend`

> **기능명세서**: FUN-005-0300 (다중 항목 추이 분석)
> **과업지시서 매핑**: 해당 없음

**Request Body**:

```json
{
  "parameters": ["aerobic_DO", "aerobic_MLSS", "influent_BOD", "effluent_TN"],
  "period": { "start": "2026-01-20", "end": "2026-02-19" },
  "max_items": 15
}
```

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "trends": {
      "aerobic_DO": { "timeseries": [], "unit": "mg/L" },
      "aerobic_MLSS": { "timeseries": [], "unit": "mg/L" },
      "influent_BOD": { "timeseries": [], "unit": "mg/L" },
      "effluent_TN": { "timeseries": [], "unit": "mg/L" }
    },
    "correlations": {
      "aerobic_DO-effluent_TN": -0.45,
      "influent_BOD-effluent_TN": 0.62
    }
  },
  "metadata": {
    "fun_id": "FUN-005-0300",
    "max_simultaneous_items": 15
  }
}
```
