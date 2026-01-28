# 008-prototype-api: API 명세서

> 하수처리장 AI 시스템 프로토타입 Mock API 명세서
> 버전: 1.0.0-prototype | 작성일: 2026-01-28

---

## 목차

1. [개요](#1-개요)
2. [공통 사항](#2-공통-사항)
3. [001-wwtp-pipeline 엔드포인트](#3-001-wwtp-pipeline-엔드포인트)
4. [002-load-removal-calc 엔드포인트](#4-002-load-removal-calc-엔드포인트)
5. [003-process-analytics 엔드포인트](#5-003-process-analytics-엔드포인트)
6. [004-anomaly-detection 엔드포인트](#6-004-anomaly-detection-엔드포인트)
7. [005-decision-support 엔드포인트](#7-005-decision-support-엔드포인트)
8. [006-control-optimization 엔드포인트](#8-006-control-optimization-엔드포인트)
9. [공통 스키마](#9-공통-스키마)

---

## 1. 개요

| 항목 | 값 |
|------|-----|
| Base URL | `http://localhost:8000` |
| API Prefix | `/api/v1` |
| 프로토콜 | HTTP (프로토타입) |
| 인증 | 없음 (프로토타입) |
| 응답 형식 | `application/json` |
| 문서 | `/docs` (Swagger UI), `/redoc` (ReDoc) |
| 헬스체크 | `GET /health` |

### 전체 엔드포인트 요약 (26개)

| 모듈 | 엔드포인트 수 | 메서드 |
|------|-------------|--------|
| 001-wwtp-pipeline | 11 | POST 10, GET 1 |
| 002-load-removal-calc | 2 | POST 1, GET 1 |
| 003-process-analytics | 3 | POST 2, GET 1 |
| 004-anomaly-detection | 3 | GET 2, POST 1 |
| 005-decision-support | 3 | GET 2, POST 1 |
| 006-control-optimization | 4 | POST 2, GET 2 |

---

## 2. 공통 사항

### 2.1 응답 헤더

모든 응답에 포함:

| 헤더 | 값 | 설명 |
|------|-----|------|
| `X-Prototype-Mode` | `"true"` | Mock API임을 표시 |
| `Content-Type` | `application/json` | 응답 형식 |

### 2.2 공통 응답 구조

**성공 응답** (200):
```json
{
  "data": { ... },
  "metadata": {
    "request_id": "req_20260127_abcdef123456",
    "timestamp": "2026-01-27T15:30:00Z",
    "processing_time_ms": 1850,
    "module_used": "001-wwtp-pipeline",
    "version": "v1"
  },
  "warnings": []
}
```

**오류 응답** (400/422/500):
```json
{
  "type": "https://api.wwtp.example.com/problems/validation-error",
  "title": "Input Validation Failed",
  "status": 422,
  "detail": "Required field '유입BOD' is missing",
  "instance": "/api/v1/predict/full-pipeline",
  "request_id": "req_20260127_xyz789",
  "timestamp": "2026-01-27T15:30:00Z",
  "errors": [
    {"field": "유입BOD", "message": "Field required", "type": "missing"}
  ]
}
```

### 2.3 Rate Limiting

| 항목 | 값 |
|------|-----|
| 방식 | 동시 요청 제한 (asyncio.Semaphore) |
| 한도 | 100 동시 요청 |
| 초과 시 | `503 Service Unavailable` + `Retry-After: 5` |

### 2.4 법적 기준 (방류수 수질)

| 항목 | 기준 (mg/L) |
|------|------------|
| BOD | ≤ 10 |
| TOC | ≤ 25 |
| SS | ≤ 10 |
| TN | ≤ 20 |
| TP | ≤ 0.2 |

---

## 3. 001-wwtp-pipeline 엔드포인트

### 3.1 공통 입력 스키마

#### PipelineInput (유입수 8개 항목)

| 필드 | 타입 | 필수 | 범위 | 단위 | 설명 |
|------|------|------|------|------|------|
| `timestamp` | string (ISO 8601) | Y | - | - | 측정 시간 |
| `유입유량` | number | Y | 50,000 ~ 200,000 | m³/일 | 유입 유량 |
| `유입BOD` | number | Y | 50 ~ 300 | mg/L | 생물화학적 산소요구량 |
| `유입TN` | number | Y | 10 ~ 50 | mg/L | 총 질소 |
| `유입TP` | number | Y | 1 ~ 10 | mg/L | 총 인 |
| `유입TOC` | number | Y | 20 ~ 100 | mg/L | 총 유기탄소 |
| `유입SS` | number | Y | 30 ~ 300 | mg/L | 부유물질 |
| `수온` | number | Y | 5 ~ 30 | ℃ | 수온 |
| `pH` | number | Y | 6 ~ 9 | - | pH |

#### OperatingConditions (운전 파라미터, 선택)

| 필드 | 단계 | 설명 |
|------|------|------|
| `일차_Q_sludge` | 02 | 1차침전지 슬러지 인발량 |
| `운전_Q_ir` | 03-04 | 내부반송 유량 |
| `운전_ir_ratio` | 03-04 | 내부반송율 |
| `운전_carbon_kg_d` | 03-04 | 탄소원 주입량 (kg/d) |
| `운전_HRT` | 03-04 | 수리학적 체류시간 |
| `송풍_Q_air` | 05 | 송풍량 |
| `송풍_n_running` | 05 | 가동 송풍기 대수 |
| `호기_DO_setpoint` | 05 | DO 설정값 (mg/L) |
| `운전_Q_ras` | 06 | 반송슬러지 유량 |
| `운전_ras_ratio` | 06 | 반송슬러지 비율 |
| `운전_Q_was` | 06 | 잉여슬러지 유량 |
| `운전_SS_was` | 06 | 잉여슬러지 SS |
| `총인처리_coag_mg_L` | 07 | 응집제 농도 (mg/L) |
| `총인처리_coag_kg_d` | 07 | 응집제 주입량 (kg/d) |
| `총인처리_Q_backwash` | 07 | 역세척 유량 |
| `HRT_total_h_in` | HRT 모델 | 총 HRT (시간) |
| `내부반송율_pct` | HRT 모델 | 내부반송율 (%) |
| `외부반송율_pct` | HRT 모델 | 외부반송율 (%) |

---

### 3.2 POST /api/v1/predict/full-pipeline

> 8단계 전체 공정 수질 예측 (FR-007)

유입수 → 1차침전 → 혐기 → 무산소 → 호기 → 2차침전 → 총인처리 → 방류의 전체 파이프라인 예측.
NEXT (t→t+1, 1시간 후) 및 HRT (t→t+13, 13시간 후) 이중 예측 결과 반환.

**Request Body**:
```json
{
  "pipeline_input": {
    "timestamp": "2024-12-31T12:00:00",
    "유입유량": 128000,
    "유입BOD": 156.8,
    "유입TN": 24.9,
    "유입TP": 4.85,
    "유입TOC": 49.31,
    "유입SS": 142.0,
    "수온": 15.5,
    "pH": 7.2
  },
  "operating_conditions": {
    "호기_DO_setpoint": 2.0,
    "운전_Q_ras": 45000,
    "총인처리_coag_mg_L": 12.5
  }
}
```

**Response** (200):
```json
{
  "data": {
    "predictions_next": {
      "stage_01_유입수": {
        "BOD_부하량_next": 20070.4,
        "TN_부하량_next": 3187.2,
        "TP_부하량_next": 620.8
      },
      "stage_02_일차침전지": {
        "일차침전_SS_eff_next": 95.2,
        "일차침전_BOD_eff_next": 125.3
      },
      "stage_03_혐기조": {
        "혐기_S_A_next": 15.2,
        "혐기_S_PO4_next": 6.8,
        "혐기_S_NH_next": 22.1,
        "혐기_MLSS_next": 2850,
        "혐기_BOD_next": 98.5
      },
      "stage_04_무산소조": {
        "무산소_S_NO_next": 3.2,
        "무산소_S_NH_next": 18.5,
        "무산소_S_A_next": 8.3,
        "무산소_MLSS_next": 2950,
        "무산소_BOD_next": 72.1
      },
      "stage_05_호기조": {
        "호기_S_NO_next": 12.5,
        "호기_S_NH_next": 1.8,
        "호기_MLSS_next": 3200,
        "호기_DO_next": 2.1,
        "호기_BOD_next": 15.3
      },
      "stage_06_이차침전지": {
        "이차침전_SS_next": 6.5,
        "이차침전_TN_next": 15.8,
        "이차침전_TP_next": 0.35,
        "이차침전_BOD_next": 8.2,
        "이차침전_COD_next": 18.5
      },
      "stage_07_총인처리": {
        "총인처리_TN_next": 15.2,
        "총인처리_TP_next": 0.12
      },
      "stage_08_방류수": {
        "방류_SS_next": 3.8,
        "방류_TN_next": 14.2,
        "방류_TP_next": 0.08,
        "방류_BOD_next": 4.5,
        "방류_COD_next": 12.8,
        "방류_NH4_next": 0.9,
        "방류_NO3_next": 11.5
      }
    },
    "predictions_hrt": {
      "stage_01_유입수": {
        "BOD_부하량_hrt": 19850.2,
        "TN_부하량_hrt": 3150.5,
        "TP_부하량_hrt": 612.3
      },
      "stage_02_일차침전지": {
        "일차침전_SS_eff_hrt": 93.5,
        "일차침전_BOD_eff_hrt": 123.8
      },
      "stage_03_혐기조": {
        "혐기_S_A_hrt": 14.8,
        "혐기_S_PO4_hrt": 7.1,
        "혐기_S_NH_hrt": 21.5,
        "혐기_MLSS_hrt": 2830,
        "혐기_BOD_hrt": 96.2
      },
      "stage_04_무산소조": {
        "무산소_S_NO_hrt": 3.0,
        "무산소_S_NH_hrt": 17.8,
        "무산소_S_A_hrt": 7.9,
        "무산소_MLSS_hrt": 2920,
        "무산소_BOD_hrt": 70.5
      },
      "stage_05_호기조": {
        "호기_S_NO_hrt": 12.1,
        "호기_S_NH_hrt": 1.5,
        "호기_MLSS_hrt": 3180,
        "호기_DO_hrt": 2.0,
        "호기_BOD_hrt": 14.8
      },
      "stage_06_이차침전지": {
        "이차침전_S_NO_hrt": 11.5,
        "이차침전_S_NH_hrt": 1.2,
        "이차침전_S_PO4_hrt": 0.3,
        "이차침전_SS_hrt": 6.2,
        "이차침전_TN_hrt": 15.2,
        "이차침전_TP_hrt": 0.32,
        "이차침전_BOD_hrt": 7.8,
        "이차침전_COD_hrt": 17.9
      },
      "stage_07_총인처리": {
        "총인처리_TN_hrt": 14.8,
        "총인처리_TP_hrt": 0.10
      },
      "stage_08_방류수": {
        "방류_SS_hrt": 3.5,
        "방류_TN_hrt": 13.8,
        "방류_TP_hrt": 0.07,
        "방류_BOD_hrt": 4.2,
        "방류_COD_hrt": 12.3,
        "방류_NH4_hrt": 0.8,
        "방류_NO3_hrt": 11.2
      }
    },
    "pipeline_output": {
      "timestamp": "2024-12-31T13:00:00",
      "방류_BOD": 4.5,
      "방류_TOC": 12.3,
      "방류_SS": 3.8,
      "방류_TN": 14.2,
      "방류_TP": 0.08,
      "compliance_check": {
        "BOD": true,
        "TOC": true,
        "SS": true,
        "TN": true,
        "TP": true
      },
      "quality_flag": "high",
      "missing_features_pct": 0.0,
      "execution_time_seconds": 0.85,
      "model_versions": {
        "stage_01": "v1.0",
        "stage_02": "v1.0",
        "stage_03": "v1.0",
        "stage_04": "v1.0",
        "stage_05": "v1.0",
        "stage_06": "v2.0",
        "stage_07": "v1.0",
        "stage_08": "v1.0"
      }
    },
    "compliance": {
      "BOD": {"limit": 10.0, "predicted": 4.5, "compliant": true, "margin": 55.0},
      "TOC": {"limit": 25.0, "predicted": 12.3, "compliant": true, "margin": 50.8},
      "SS": {"limit": 10.0, "predicted": 3.8, "compliant": true, "margin": 62.0},
      "TN": {"limit": 20.0, "predicted": 14.2, "compliant": true, "margin": 29.0},
      "TP": {"limit": 0.2, "predicted": 0.08, "compliant": true, "margin": 60.0}
    }
  },
  "metadata": {
    "request_id": "req_20260127_abcdef123456",
    "timestamp": "2026-01-27T15:30:00Z",
    "processing_time_ms": 1850,
    "module_used": "001-wwtp-pipeline",
    "version": "v1",
    "missing_handled": {
      "method": "none",
      "missing_fields": [],
      "imputation_details": ""
    },
    "model_performance": {
      "MAPE": 15.3,
      "R2": 0.92,
      "degraded": false
    }
  },
  "warnings": []
}
```
---

### 3.3 POST /api/v1/simulate/parameter-change

> 운영인자 변경 시나리오 시뮬레이션

최대 5개 시나리오 동시 비교. 기준 조건(PipelineInput + 제어인자) 대비 제어인자 변경 시 방류수 수질(Stage08OutputHRT) 변화 예측.

**Request Body**:
```json
{
  "base_conditions": {
    "pipeline_input": {
      "timestamp": "2024-12-31T12:00:00",
      "유입유량": 128000,
      "유입BOD": 156.8,
      "유입TN": 24.9,
      "유입TP": 4.85,
      "유입TOC": 49.31,
      "유입SS": 142.0,
      "수온": 15.5,
      "pH": 7.2
    },
    "control_params": {
      "송풍량": 85000,
      "내부반송율": 300,
      "외부탄소원": 0,
      "응집제주입": 12.5,
      "호기_MLSS": 3200
    }
  },
  "scenarios": [
    {
      "scenario_name": "송풍량 증가",
      "changes": {
        "송풍량": 95000
      }
    },
    {
      "scenario_name": "MLSS 3500 + 응집제 증량",
      "changes": {
        "호기_MLSS": 3500,
        "응집제주입": 18.0
      }
    }
  ]
}
```

**SimulationControlParams (제어인자 5개)**:

| 필드 | 타입 | 범위 | 단위 | 설명 |
|------|------|------|------|------|
| `송풍량` | number | 30,000~150,000 | m³/h | 호기조 송풍량 |
| `내부반송율` | number | 100~500 | % | 내부반송율 |
| `외부탄소원` | number | 0~500 | kg/d | 외부탄소원 주입량 |
| `응집제주입` | number | 0~50 | mg/L | 응집제 주입량 |
| `호기_MLSS` | number | 1,500~5,000 | mg/L | 호기조 MLSS 농도 |

**Response** (200):
```json
{
  "data": {
    "base_result": {
      "방류_SS_hrt": 3.5,
      "방류_TN_hrt": 13.8,
      "방류_TP_hrt": 0.07,
      "방류_BOD_hrt": 4.2,
      "방류_COD_hrt": 12.3,
      "방류_NH4_hrt": 0.8,
      "방류_NO3_hrt": 11.2
    },
    "scenario_results": [
      {
        "scenario_name": "송풍량 증가",
        "prediction": {
          "방류_SS_hrt": 3.2,
          "방류_TN_hrt": 12.5,
          "방류_TP_hrt": 0.06,
          "방류_BOD_hrt": 3.8,
          "방류_COD_hrt": 11.5,
          "방류_NH4_hrt": 0.5,
          "방류_NO3_hrt": 10.8
        },
        "is_compliant": true
      },
      {
        "scenario_name": "MLSS 3500 + 응집제 증량",
        "prediction": {
          "방류_SS_hrt": 3.0,
          "방류_TN_hrt": 13.2,
          "방류_TP_hrt": 0.04,
          "방류_BOD_hrt": 3.5,
          "방류_COD_hrt": 11.0,
          "방류_NH4_hrt": 0.6,
          "방류_NO3_hrt": 11.0
        },
        "is_compliant": true
      }
    ],
    "recommended_scenario": "MLSS 3500 + 응집제 증량"
  },
  "metadata": { ... },
  "warnings": []
}
```

---

## 4. 002-load-removal-calc 엔드포인트

### 4.1 POST /api/v1/load/calculate

> 부하량 및 제거율 계산 (FR-025)

유입유량과 수질 데이터를 입력하고, 측정 대상 탱크(유입수/혐기조/무산소조/호기조)를 지정하면 해당 지점의 BOD, TN, TP, SS 부하량 및 제거율을 반환.

**Request Body**:
```json
{
  "target_tank": "influent",
  "flow_rate": 128000,
  "influent": {
    "BOD": 156.8,
    "TN": 24.9,
    "TP": 4.85,
    "SS": 142.0
  },
  "effluent": {
    "BOD": 4.5,
    "TN": 14.2,
    "TP": 0.08,
    "SS": 3.8
  }
}
```

**Request 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `target_tank` | string | O | 측정 대상 탱크: `influent` (유입수), `anaerobic` (혐기조), `anoxic` (무산소조), `aerobic` (호기조) |
| `flow_rate` | number | O | 유입유량 (m³/day) |
| `influent` | object | O | 유입 수질 데이터 (BOD, TN, TP, SS - mg/L) |
| `effluent` | object | X | 유출 수질 데이터 (제거율 계산용, 선택) |

**Response** (200):
```json
{
  "data": {
    "target_tank": "influent",
    "flow_rate": 128000,
    "loads": {
      "BOD_load_kg_d": 20070.4,
      "TN_load_kg_d": 3187.2,
      "TP_load_kg_d": 620.8,
      "SS_load_kg_d": 18176.0
    },
    "removal_rates": {
      "BOD_removal_pct": 97.1,
      "TN_removal_pct": 43.0,
      "TP_removal_pct": 98.4,
      "SS_removal_pct": 97.3
    },
    "timestamp": "2026-01-27T14:30:00Z"
  },
  "metadata": { ... }
}
```

### 4.2 GET /api/v1/load/statistics

> 부하량 통계 조회 (FR-026)

**Query Parameters**: `start_date`, `end_date`, `interval` (daily/weekly/monthly), `target_tank` (influent/anaerobic/anoxic/aerobic, 선택)

**Response** (200):
```json
{
  "data": {
    "period": { "start": "2026-01-01", "end": "2026-01-31", "interval": "daily" },
    "target_tank": "influent",
    "statistics": [
      {
        "date": "2026-01-27",
        "BOD_load_avg": 19500.5,
        "TN_load_avg": 3100.2,
        "TP_load_avg": 600.5,
        "removal_rate_BOD_avg": 96.8,
        "removal_rate_TN_avg": 72.5,
        "removal_rate_TP_avg": 83.0
      }
    ]
  },
  "metadata": { ... }
}
```

---

## 5. 003-process-analytics 엔드포인트

> 규칙 기반 공정 지표 계산 (AI 모델 불필요). 9개 지표: 평균DO, MLSS, F/M비, C/N비, 질산화율, SNR, SDNR, 침전지 수면적부하, 침전시간.

### 5.1 POST /api/v1/analytics/indicators/all

> 전체 공정 지표 산출 (FR-027)

운전 데이터를 입력하면 9개 공정 지표를 한번에 계산. 각 지표에 정상 범위 초과 여부(alert) 포함.

**Request Body**:
```json
{
  "flow_rate": 128000,
  "influent": { "BOD": 156.8, "TN": 24.9, "NH4": 18.5 },
  "effluent": { "BOD": 4.5, "TN": 14.2, "NH4": 1.2, "NO3": 10.5 },
  "aerobic": { "DO": 2.3, "MLSS": 3200, "volume": 25000 },
  "anoxic": { "MLSS": 3000, "volume": 15000, "influent_NO3": 12.0, "effluent_NO3": 2.5 },
  "sedimentation": { "surface_area": 5200, "depth": 3.5 }
}
```

**Request 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `flow_rate` | number | O | 유입유량 (m³/day) |
| `influent` | object | O | 유입 수질 (BOD, TN, NH4 - mg/L) |
| `effluent` | object | O | 유출 수질 (BOD, TN, NH4, NO3 - mg/L) |
| `aerobic` | object | O | 호기조 데이터 (DO mg/L, MLSS mg/L, volume m³) |
| `anoxic` | object | X | 무산소조 데이터 (SDNR 계산용) |
| `sedimentation` | object | O | 침전지 데이터 (surface_area m², depth m) |

**Response** (200):
```json
{
  "data": {
    "indicators": {
      "avg_do":                { "value": 2.3,   "unit": "mg/L",              "normal_range": { "min": 1.5, "max": 3.0 },  "alert": false, "formula": "호기조 DO 평균" },
      "mlss":                  { "value": 3200,  "unit": "mg/L",              "normal_range": { "min": 2000, "max": 4000 }, "alert": false, "formula": "호기조 MLSS 농도" },
      "fm_ratio":              { "value": 0.25,  "unit": "kg BOD/kg MLSS·d",  "normal_range": { "min": 0.2, "max": 0.4 },  "alert": false, "formula": "(Q × BOD_inf) / (V_aerobic × MLSS)" },
      "cn_ratio":              { "value": 6.3,   "unit": "-",                 "normal_range": { "min": 4.0, "max": 999 },  "alert": false, "formula": "BOD_inf / TN_inf" },
      "nitrification_rate":    { "value": 93.5,  "unit": "%",                 "normal_range": { "min": 90, "max": 100 },   "alert": false, "formula": "(NH4_inf - NH4_eff) / NH4_inf × 100" },
      "snr":                   { "value": 0.054, "unit": "kg NH4-N/kg MLSS·d","normal_range": { "min": 0.03, "max": 0.08 },"alert": false, "formula": "Q × (NH4_inf - NH4_eff) / (V_aerobic × MLSS)" },
      "sdnr":                  { "value": 0.048, "unit": "kg NO3-N/kg MLSS·d","normal_range": { "min": 0.03, "max": 0.07 },"alert": false, "formula": "Q × (NO3_in - NO3_out) / (V_anoxic × MLSS_anoxic)" },
      "surface_overflow_rate": { "value": 24.6,  "unit": "m³/m²·d",          "normal_range": { "min": 0, "max": 30 },     "alert": false, "formula": "Q / A_sedimentation" },
      "settling_time":         { "value": 2.8,   "unit": "h",                 "normal_range": { "min": 1.5, "max": 4.0 },  "alert": false, "formula": "(A × H) / (Q / 24)" }
    },
    "alert_count": 0,
    "timestamp": "2026-01-27T14:30:00Z"
  },
  "metadata": { ... }
}
```

**지표 계산 공식**:

| 지표 | 공식 | 단위 | 정상 범위 |
|------|------|------|-----------|
| `avg_do` | 호기조 DO 평균 | mg/L | 1.5~3.0 |
| `mlss` | 호기조 MLSS 농도 | mg/L | 2,000~4,000 |
| `fm_ratio` | (Q × BOD_inf) / (V_aerobic × MLSS) | kg BOD/kg MLSS·d | 0.2~0.4 |
| `cn_ratio` | BOD_inf / TN_inf | - | ≥4.0 |
| `nitrification_rate` | (NH4_inf - NH4_eff) / NH4_inf × 100 | % | ≥90 |
| `snr` | Q × (NH4_inf - NH4_eff) / (V_aerobic × MLSS) | kg NH4-N/kg MLSS·d | 0.03~0.08 |
| `sdnr` | Q × (NO3_in - NO3_out) / (V_anoxic × MLSS_anoxic) | kg NO3-N/kg MLSS·d | 0.03~0.07 |
| `surface_overflow_rate` | Q / A_sedimentation | m³/m²·d | ≤30 |
| `settling_time` | (A × H) / (Q / 24) | h | 1.5~4.0 |

### 5.2 POST /api/v1/analytics/indicators/select

> 선택 공정 지표 산출 (FR-028)

요청한 지표만 계산하여 반환. `indicators` 배열로 지정.

**Request Body**:
```json
{
  "flow_rate": 128000,
  "influent": { "BOD": 156.8, "TN": 24.9, "NH4": 18.5 },
  "effluent": { "BOD": 4.5, "TN": 14.2, "NH4": 1.2, "NO3": 10.5 },
  "aerobic": { "DO": 2.3, "MLSS": 3200, "volume": 25000 },
  "sedimentation": { "surface_area": 5200 },
  "indicators": ["fm_ratio", "cn_ratio", "nitrification_rate"]
}
```

**indicators enum**: `avg_do`, `mlss`, `fm_ratio`, `cn_ratio`, `nitrification_rate`, `snr`, `sdnr`, `surface_overflow_rate`, `settling_time`

**Response** (200):
```json
{
  "data": {
    "requested": ["fm_ratio", "cn_ratio", "nitrification_rate"],
    "indicators": {
      "fm_ratio":           { "value": 0.25, "unit": "kg BOD/kg MLSS·d", "normal_range": { "min": 0.2, "max": 0.4 }, "alert": false, "formula": "(Q × BOD_inf) / (V_aerobic × MLSS)" },
      "cn_ratio":           { "value": 6.3,  "unit": "-",                "normal_range": { "min": 4.0, "max": 999 }, "alert": false, "formula": "BOD_inf / TN_inf" },
      "nitrification_rate": { "value": 93.5, "unit": "%",                "normal_range": { "min": 90, "max": 100 },  "alert": false, "formula": "(NH4_inf - NH4_eff) / NH4_inf × 100" }
    },
    "alert_count": 0,
    "timestamp": "2026-01-27T14:30:00Z"
  },
  "metadata": { ... }
}
```

---

## 6. 004-anomaly-detection 엔드포인트

- 추가 분석 필요

---

## 7. 005-decision-support 엔드포인트

- 추가 분석 필요

---

## 8. 006-control-optimization 엔드포인트

### 8.1 POST /api/v1/control/command

> 최적 제어인자 도출 (FR-036)

유입수질과 방류수질 목표를 입력하면, 전체 공정 시뮬레이션(피드포워드)을 통해 목표를 만족시키는 최적 제어인자(송풍량, 내부반송율, 외부탄소원, 응집제주입, 호기_MLSS, 호기_DO)와 예상 방류수질(BOD, COD, TN, TP, SS, NH4)을 반환.

**Request Body**:
```json
{
  "execution_mode": "recommendation",
  "influent": {
    "flow_rate": 128000,
    "BOD": 156.8,
    "COD": 98.5,
    "TN": 24.9,
    "TP": 4.85,
    "temperature": 18.5,
    "pH": 7.2
  },
  "effluent_targets": {
    "BOD": 10.0,
    "COD": 40.0,
    "TN": 20.0,
    "TP": 0.2
  }
}
```

**influent (유입수질 데이터)**:

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `flow_rate` | number (required) | m³/day | 유입유량 |
| `BOD` | number (required) | mg/L | 유입 BOD |
| `COD` | number (required) | mg/L | 유입 COD |
| `TN` | number (required) | mg/L | 유입 TN |
| `TP` | number (required) | mg/L | 유입 TP |
| `temperature` | number (required) | °C | 수온 |
| `pH` | number (required) | - | pH |

**effluent_targets (방류수질 목표)**:

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `BOD` | number (required) | mg/L | 방류 BOD 목표 |
| `COD` | number (required) | mg/L | 방류 COD 목표 |
| `TN` | number (required) | mg/L | 방류 TN 목표 |
| `TP` | number (required) | mg/L | 방류 TP 목표 |

**Response** (200):
```json
{
  "data": {
    "command_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-01-27T14:30:00Z",
    "execution_mode": "recommendation",
    "status": "completed",
    "optimized_controls": {
      "송풍량": 72000,
      "내부반송율": 280,
      "외부탄소원": 50,
      "응집제주입": 15.0,
      "호기_MLSS": 3400,
      "호기_DO": 2.5
    },
    "predicted_effluent": {
      "BOD": 4.2,
      "COD": 12.3,
      "TN": 13.8,
      "TP": 0.07,
      "SS": 3.5,
      "NH4": 0.8
    },
    "energy_cost_reduction_pct": 12.8,
    "result": "success"
  },
  "metadata": { ... }
}
```

**optimized_controls (최적 제어인자)**:

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `송풍량` | number | m³/h | 호기조 송풍량 |
| `내부반송율` | number | % | 내부반송율 |
| `외부탄소원` | number | kg/d | 외부탄소원 주입량 |
| `응집제주입` | number | mg/L | 응집제 주입량 |
| `호기_MLSS` | number | mg/L | 호기조 MLSS 농도 |
| `호기_DO` | number | mg/L | 호기조 DO 설정값 |

**predicted_effluent (예상 방류수질)**:

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `BOD` | number | mg/L | 예상 방류 BOD |
| `COD` | number | mg/L | 예상 방류 COD |
| `TN` | number | mg/L | 예상 방류 TN |
| `TP` | number | mg/L | 예상 방류 TP |
| `SS` | number | mg/L | 예상 방류 SS |
| `NH4` | number | mg/L | 예상 방류 NH4 |



---

## 9. 공통 스키마

### 9.1 단계별 출력 필드 요약

#### NEXT 모델 (t→t+1, 1시간 후 예측) - 실시간 제어용

| 단계 | 스키마 | 출력 필드 |
|------|--------|----------|
| 01 유입수 | Stage01OutputNEXT | BOD_부하량_next, TN_부하량_next, TP_부하량_next |
| 02 1차침전 | Stage02OutputNEXT | 일차침전_SS_eff_next, 일차침전_BOD_eff_next |
| 03 혐기조 | Stage03OutputNEXT | 혐기_S_A_next, 혐기_S_PO4_next, 혐기_S_NH_next, 혐기_MLSS_next, 혐기_BOD_next |
| 04 무산소조 | Stage04OutputNEXT | 무산소_S_NO_next, 무산소_S_NH_next, 무산소_S_A_next, 무산소_MLSS_next, 무산소_BOD_next |
| 05 호기조 | Stage05OutputNEXT | 호기_S_NO_next, 호기_S_NH_next, 호기_MLSS_next, 호기_DO_next, 호기_BOD_next |
| 06 2차침전 | Stage06OutputNEXT | 이차침전_SS_next, 이차침전_TN_next, 이차침전_TP_next, 이차침전_BOD_next, 이차침전_COD_next |
| 07 총인처리 | Stage07OutputNEXT | 총인처리_TN_next, 총인처리_TP_next |
| 08 방류수 | Stage08OutputNEXT | 방류_SS_next, 방류_TN_next, 방류_TP_next, 방류_BOD_next, 방류_COD_next, 방류_NH4_next, 방류_NO3_next |

#### HRT 모델 (t→t+13, 13시간 후 예측) - 공정 계획용

| 단계 | 스키마 | 출력 필드 |
|------|--------|----------|
| 01 유입수 | Stage01OutputHRT | BOD_부하량_hrt, TN_부하량_hrt, TP_부하량_hrt |
| 02 1차침전 | Stage02OutputHRT | 일차침전_SS_eff_hrt, 일차침전_BOD_eff_hrt |
| 03 혐기조 | Stage03OutputHRT | 혐기_S_A_hrt, 혐기_S_PO4_hrt, 혐기_S_NH_hrt, 혐기_MLSS_hrt, 혐기_BOD_hrt |
| 04 무산소조 | Stage04OutputHRT | 무산소_S_NO_hrt, 무산소_S_NH_hrt, 무산소_S_A_hrt, 무산소_MLSS_hrt, 무산소_BOD_hrt |
| 05 호기조 | Stage05OutputHRT | 호기_S_NO_hrt, 호기_S_NH_hrt, 호기_MLSS_hrt, 호기_DO_hrt, 호기_BOD_hrt |
| 06 2차침전 | Stage06OutputHRT | 이차침전_S_NO_hrt, 이차침전_S_NH_hrt, 이차침전_S_PO4_hrt, 이차침전_SS_hrt, 이차침전_TN_hrt, 이차침전_TP_hrt, 이차침전_BOD_hrt, 이차침전_COD_hrt |
| 07 총인처리 | Stage07OutputHRT | 총인처리_TN_hrt, 총인처리_TP_hrt |
| 08 방류수 | Stage08OutputHRT | 방류_SS_hrt, 방류_TN_hrt, 방류_TP_hrt, 방류_BOD_hrt, 방류_COD_hrt, 방류_NH4_hrt, 방류_NO3_hrt |

### 9.2 PipelineOutput

| 필드 | 타입 | 설명 |
|------|------|------|
| `timestamp` | datetime | 예측 시간 (입력 + 1시간) |
| `방류_BOD` | number | mg/L |
| `방류_TOC` | number | mg/L |
| `방류_SS` | number | mg/L |
| `방류_TN` | number | mg/L |
| `방류_TP` | number | mg/L |
| `compliance_check` | object | 항목별 법적 기준 준수 (boolean) |
| `quality_flag` | enum | high / medium / low / degraded |
| `missing_features_pct` | number | 결측 피처 비율 (%) |
| `execution_time_seconds` | number | 파이프라인 실행 시간 (초) |
| `model_versions` | object | 단계별 모델 버전 |

### 9.3 ComplianceCheck

| 필드 | 타입 | 설명 |
|------|------|------|
| `limit` | number | 법적 기준 (mg/L) |
| `predicted` | number | 예측값 (mg/L) |
| `compliant` | boolean | 기준 준수 여부 |
| `margin` | number | 여유율 (%) |

### 9.4 ResponseMetadata

| 필드 | 타입 | 설명 |
|------|------|------|
| `request_id` | string | 요청 ID |
| `timestamp` | datetime | 응답 생성 시간 |
| `processing_time_ms` | integer | 처리 시간 (ms) |
| `module_used` | string | 사용 모듈 |
| `version` | string | API 버전 |
| `missing_handled` | object | 센서 결측 처리 (method, missing_fields, imputation_details) |
| `model_performance` | object | 모델 성능 (MAPE, R2, degraded) |

### 9.5 Quality Flag 기준

| 등급 | 조건 | 설명 |
|------|------|------|
| `high` | 결측 0% | 모든 피처 사용 |
| `medium` | 결측 < 10% | 일부 결측, 대체값 적용 |
| `low` | 결측 10-50% | 다수 결측, 예측 신뢰도 저하 |
| `degraded` | 결측 > 50% | 과반 결측, 예측 품질 보장 불가 |

---

**OpenAPI 계약 파일**: [contracts/wwtp_pipeline.yaml](contracts/wwtp_pipeline.yaml)
**데이터 모델**: [data-model.md](data-model.md)
**기능 명세**: [spec.md](spec.md)

**작성일**: 2026-01-28
