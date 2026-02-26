# MFR-005: 공정별 예측 모델 (공정시뮬레이션 포함)

- **과업지시서**: MFR-005 — 대상 처리공정별 선정 모델을 통한 수질예측, 운영자가 설정한 유입성상·유량·운영조건 변동에 대한 공정시뮬레이션
- **데이터정의서 모델**: MFR-005-1a ~ MFR-005-4b (8개 서브모델)
- **기능명세서**: FUN-001-0100 ~ FUN-001-0500 (공정 감시), FUN-004-0200 (방류수 예측), FUN-000-0100 (통합 상황판)
- **Spec 모듈**: `001-wwtp-pipeline`
- **성능 목표**: 오차율 평가, RMSE·MAE·R² 참조

---

## 설계 기준 참조

> 공정 예측 모델의 입력 범위 기준입니다.

**유입수질 입력 범위** (databook 기준):

| 항목 | 범위 | 단위 |
|------|------|------|
| 유입유량 (Q) | 60,000 ~ 170,000 | m³/d |
| BOD | 100 ~ 300 | mg/L |
| COD | 160 ~ 480 | mg/L |
| TN | 20 ~ 60 | mg/L |
| TP | 2 ~ 8 | mg/L |
| SS | 150 ~ 400 | mg/L |
| TOC | 50 ~ 150 | mg/L |
| 수온 | 10 ~ 30 | ℃ |
| pH | 6.5 ~ 8.5 | - |

**운전 파라미터 범위**:

| 항목 | 설계값 | 범위 | 단위 |
|------|--------|------|------|
| MLSS | 3,500 | - | mg/L |
| DO 설정 | 2.0 | 1.5 ~ 3.5 | mg/L |
| RAS 비율 | 30% | 25 ~ 35% | - |
| IR 비율 | 동적제어 | 100 ~ 400% | - |
| 외부탄소원 | 동적제어 | 0 ~ 20,000 | kg/d |
| 응집제 (PAC) | 30 | 30 ~ 100 | mg/L |
| Q_was | 동적제어 | 500 ~ 8,000 | m³/d |

**방류수 법적 기준** (I지역):

| 항목 | 기준 | 단위 |
|------|------|------|
| BOD | ≤ 5 | mg/L |
| COD | ≤ 40 | mg/L |
| SS | ≤ 10 | mg/L |
| TN | ≤ 20 | mg/L |
| TP | ≤ 0.2 | mg/L |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

---

## 공통 스키마

### InfluentData

유입수 데이터 입력 스키마입니다. `flow`, `BOD`, `COD`, `TN`, `TP`는 필수이며, 나머지는 선택입니다.

> **소스**: `api/schemas.py` — `class InfluentData(BaseModel)`

| 필드 | 타입 | 필수 | 검증 | 단위 | 설명 |
|------|------|------|------|------|------|
| `flow` | float | Y | `gt=0` | m³/day | 유입유량 |
| `BOD` | float | Y | `ge=0` | mg/L | 유입 BOD |
| `COD` | float | Y | `ge=0` | mg/L | 유입 COD |
| `TN` | float | Y | `ge=0` | mg/L | 유입 TN |
| `TP` | float | Y | `ge=0` | mg/L | 유입 TP |
| `SS` | float | N | `ge=0` | mg/L | 유입 SS |
| `TOC` | float | N | `ge=0` | mg/L | 유입 TOC |
| `temperature` | float | N | - | ℃ | 수온 |
| `pH` | float | N | `ge=0, le=14` | - | pH |

### OperationData

운전 조작 변수 스키마입니다. 모든 필드가 선택입니다.

> **소스**: `api/schemas.py` — `class OperationData(BaseModel)`

| 필드 | 타입 | 필수 | 검증 | 단위 | 설명 |
|------|------|------|------|------|------|
| `MLSS` | float | N | `ge=0` | mg/L | 호기조 MLSS |
| `DO_setpoint` | float | N | `ge=0` | mg/L | DO 설정값 |
| `RAS_rate` | float | N | `ge=0` | % | 외부반송율 |
| `IR_rate` | float | N | `ge=0` | % | 내부반송율 |
| `carbon_dose` | float | N | `ge=0` | L/day | 외부탄소원 투입량 |
| `coagulant_dose` | float | N | `ge=0` | L/day | 응집제 투입량 |
| `WAS_volume` | float | N | `ge=0` | m³/day | 잉여슬러지 인발량 |
| `airflow` | float | N | `ge=0` | m³/hr | 송풍량 |

---

## 서비스 레이어

### PredictionService

> **소스**: `api/services/prediction_service.py`

`PredictionService`는 `wwtp_pipeline` Stage 모듈을 래핑하여 예측을 수행하는 서비스입니다. Lazy loading으로 모델을 초기화합니다.

**Stage 설정 매핑 (STAGE_CONFIG)**:

| stage (path param) | Stage 클래스 | 모델 ID | 모델 경로 | 공정명 |
|---------------------|-------------|---------|-----------|--------|
| `influent` | `Stage01Influent` | MFR-005-1a | `models/01_유입수/` | 유입수 |
| `primary-clarifier` | `Stage02PrimaryClarifier` | MFR-005-1b | `models/02_일차침전지/` | 일차침전지 |
| `anaerobic` | `Stage03Anaerobic` | MFR-005-2a | `models/03_혐기조/` | 혐기조 |
| `anoxic` | `Stage04Anoxic` | MFR-005-2b | `models/04_무산소조/` | 무산소조 |
| `aerobic` | `Stage05Aerobic` | MFR-005-2c | `models/05_호기조/` | 호기조 |
| `secondary-clarifier` | `Stage06SecondaryClarifier` | MFR-005-3a | `models/06_이차침전지/` | 이차침전지 |
| `tp-treatment` | `Stage07TotalPhosphorus` | MFR-005-4b | `models/07_총인처리/` | 총인처리 |
| `effluent` | `Stage08Effluent` | MFR-005-4a | `models/08_방류수/` | 방류수 |

---

## 1. 전체 파이프라인 예측

### `POST /predict/full-pipeline`

유입수 데이터를 입력받아 8단계 전체 공정의 수질을 순차 예측합니다.

> **구현 상태**: `NotImplementedError` (PipelineOrchestrator 미구현)

**Request Body** (`FullPipelineRequest`):

```json
{
  "influent": {
    "flow": 10000,
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "SS": 120.0,
    "TOC": 80.0,
    "temperature": 18.5,
    "pH": 7.2
  },
  "operation": {
    "MLSS": 3500,
    "DO_setpoint": 2.0,
    "RAS_rate": 50,
    "IR_rate": 200,
    "carbon_dose": 100,
    "coagulant_dose": 50,
    "WAS_volume": 200,
    "airflow": 5000
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `influent` | InfluentData | Y | 유입수 데이터 (상세: 공통 스키마 참조) |
| `operation` | OperationData | N | 운전 조작 변수 (상세: 공통 스키마 참조) |

**Response Body** (`200 OK`, `FullPipelineResponse`):

```json
{
  "success": true,
  "data": {
    "stages": {
      "influent": {
        "model_id": "MFR-005-1a",
        "predictions": {
          "flow": { "value": 10200, "unit": "m³/day" },
          "BOD": { "value": 155.0, "unit": "mg/L" },
          "TN": { "value": 36.0, "unit": "mg/L" },
          "TP": { "value": 5.2, "unit": "mg/L" }
        }
      },
      "primary_clarifier": {
        "model_id": "MFR-005-1b",
        "predictions": {
          "SS": { "value": 54.0, "unit": "mg/L" },
          "BOD": { "value": 105.0, "unit": "mg/L" },
          "sludge": { "value": 792.0, "unit": "kg/day" }
        }
      },
      "anaerobic": {
        "model_id": "MFR-005-2a",
        "predictions": {
          "P_release": { "value": 8.5, "unit": "mg/L" },
          "VFA_consumed": { "value": 45.0, "unit": "mg/L" }
        }
      },
      "anoxic": {
        "model_id": "MFR-005-2b",
        "predictions": {
          "denitrification": { "value": 6.2, "unit": "mg/L" },
          "NO3N": { "value": 2.3, "unit": "mg/L" }
        }
      },
      "aerobic": {
        "model_id": "MFR-005-2c",
        "predictions": {
          "NH4N": { "value": 1.1, "unit": "mg/L" },
          "P_uptake": { "value": 7.8, "unit": "mg/L" },
          "delta_MLSS": { "value": 50, "unit": "mg/L" }
        }
      },
      "secondary_clarifier": {
        "model_id": "MFR-005-3a",
        "predictions": {
          "settling_efficiency": { "value": 98.5, "unit": "%" },
          "RAS_concentration": { "value": 7200, "unit": "mg/L" }
        }
      },
      "tp_treatment": {
        "model_id": "MFR-005-4b",
        "predictions": {
          "TP": { "value": 0.15, "unit": "mg/L" }
        }
      },
      "effluent": {
        "model_id": "MFR-005-4a",
        "predictions": {
          "BOD": { "value": 4.5, "unit": "mg/L" },
          "COD": { "value": 14.2, "unit": "mg/L" },
          "TN": { "value": 13.8, "unit": "mg/L" },
          "SS": { "value": 3.2, "unit": "mg/L" }
        }
      }
    },
    "compliance": {
      "BOD": { "predicted": 4.5, "limit": 5, "status": "pass", "margin": 10.0 },
      "COD": { "predicted": 14.2, "limit": 40, "status": "pass", "margin": 64.5 },
      "TN": { "predicted": 13.8, "limit": 20, "status": "pass", "margin": 31.0 },
      "TP": { "predicted": 0.15, "limit": 0.2, "status": "pass", "margin": 25.0 },
      "SS": { "predicted": 3.2, "limit": 10, "status": "pass", "margin": 68.0 }
    },
    "warnings": []
  },
  "metadata": {
    "model_id": "MFR-005",
    "pipeline_version": "1.0",
    "total_stages": 8,
    "predicted_at": "2026-02-19T10:00:00+09:00"
  }
}
```

**에러 응답**:

| HTTP 코드 | 설명 |
|-----------|------|
| `422` | 입력 데이터 검증 실패 (Pydantic ValidationError) |
| `500` | 파이프라인 내부 오류 |

---

## 2. 개별 공정 예측

### `POST /predict/{stage}`

개별 공정 단계의 수질을 예측합니다.

> **구현 상태**: `PredictionService` 연동 완료 (일부 Stage 테스트 진행 중)
> **소스**: `wwtp_pipeline/stages/stage_01_influent.py` ~ `stage_08_effluent.py`

**Path Parameters** (`StageEnum`):

| stage | 모델 ID | 설명 |
|-------|--------|------|
| `influent` | MFR-005-1a | 유입수 예측 |
| `primary-clarifier` | MFR-005-1b | 1차침전지 예측 |
| `anaerobic` | MFR-005-2a | 혐기조 예측 |
| `anoxic` | MFR-005-2b | 무산소조 예측 |
| `aerobic` | MFR-005-2c | 호기조 예측 |
| `secondary-clarifier` | MFR-005-3a | 2차침전지 예측 |
| `tp-treatment` | MFR-005-4b | 총인처리 예측 |
| `effluent` | MFR-005-4a | 방류수 예측 |

**Request Body** (`StagePredictRequest`):

```json
{
  "stage_input": { ... },
  "upstream_output": { ... }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `stage_input` | dict | Y | 공정별 입력 데이터. 값이 list인 경우 시계열로 처리 |
| `upstream_output` | dict | N | 상류 공정 출력 데이터 (자동 병합) |

> **시계열 데이터**: 각 Stage의 FeatureEngineer는 lag/rolling 피처를 위해 시계열 데이터가 필요합니다. `stage_input`의 값을 list로 전달하면 시계열로 처리됩니다. 단일 값(scalar) 전달 시 `min_data_length`만큼 자동 복제됩니다.

**공통 에러 응답**:

| HTTP 코드 | 설명 |
|-----------|------|
| `422` | 입력 데이터 검증 실패 또는 잘못된 stage 값 |
| `500` | 모델 로딩 실패 또는 예측 오류 |

---

### 2.1 유입수 (`influent`)

> **소스**: `wwtp_pipeline/stages/stage_01_influent.py` — `Stage01Influent`, `InflowFeatureEngineer`

유입수 6개 항목의 시계열 데이터를 입력받아 BOD/TN/TP **부하량**(kg/일)을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-1a |
| **min_data_length** | 169 (lag 168h + 1행) |
| **Lag 피처** | 6컬럼 × 2 lag (24h, 168h) = 12개 |
| **Rolling 피처** | 4컬럼 × 4 통계 (mean, std, max, min) × 24h = 16개 |

**필수 입력 컬럼** (6개):

| 컬럼명 | 단위 | 설명 |
|--------|------|------|
| `유입_Q` | m³/d | 유입유량 |
| `유입_BOD` | mg/L | 유입 BOD |
| `유입_TN` | mg/L | 유입 TN |
| `유입_TP` | mg/L | 유입 TP |
| `유입_TOC` | mg/L | 유입 TOC |
| `유입_SS` | mg/L | 유입 SS |

**출력 스키마** (`Stage01Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `BOD_부하량` | float | kg/일 | BOD 부하량 (유량 × 농도 × 1e-3) |
| `TN_부하량` | float | kg/일 | TN 부하량 |
| `TP_부하량` | float | kg/일 | TP 부하량 |

**Request 예시** (`POST /predict/influent`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, "...(169개 시계열)"],
    "유입_BOD": [150.0, 152.0, "..."],
    "유입_TN": [35.0, 34.5, "..."],
    "유입_TP": [5.0, 5.1, "..."],
    "유입_TOC": [80.0, 82.0, "..."],
    "유입_SS": [120.0, 118.0, "..."]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "BOD_부하량": 15000.0,
      "TN_부하량": 3500.0,
      "TP_부하량": 500.0
    },
    "model_id": "MFR-005-1a",
    "stage_name": "유입수"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.2 일차침전지 (`primary-clarifier`)

> **소스**: `wwtp_pipeline/stages/stage_02_primary_clarifier.py` — `Stage02PrimaryClarifier`, `PrimaryClarifierFeatureEngineer`

유입수 8개 항목의 시계열 데이터를 기반으로 일차침전지 유출수 BOD/SS를 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-1b |
| **min_data_length** | 168 (초기 168행 제거 후 dropna) |
| **Lag 피처** | 6컬럼 × 4 lag (1h, 2h, 3h, 6h) = 24개 |
| **Rolling 피처** | 4컬럼 × 4 통계 (mean, std, max, min) × 24h = 16개 |

**필수 입력 컬럼** (8개):

| 컬럼명 | 단위 | 설명 |
|--------|------|------|
| `유입_Q` | m³/d | 유입유량 |
| `유입_BOD` | mg/L | 유입 BOD |
| `유입_TN` | mg/L | 유입 TN |
| `유입_TP` | mg/L | 유입 TP |
| `유입_TOC` | mg/L | 유입 TOC |
| `유입_SS` | mg/L | 유입 SS |
| `유입_T` | ℃ | 수온 |
| `유입_pH` | - | pH |

**출력 스키마** (`Stage02Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `일차침전_BOD_eff_next` | float | mg/L | 일차침전지 유출 BOD 예측값 (t+1) |
| `일차침전_SS_eff_next` | float | mg/L | 일차침전지 유출 SS 예측값 (t+1) |

**Request 예시** (`POST /predict/primary-clarifier`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, "...(168개 시계열)"],
    "유입_BOD": [150.0, 152.0, "..."],
    "유입_TN": [35.0, 34.5, "..."],
    "유입_TP": [5.0, 5.1, "..."],
    "유입_TOC": [80.0, 82.0, "..."],
    "유입_SS": [120.0, 118.0, "..."],
    "유입_T": [18.0, 18.0, "..."],
    "유입_pH": [7.2, 7.2, "..."]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "일차침전_BOD_eff_next": 125.3,
      "일차침전_SS_eff_next": 95.2
    },
    "model_id": "MFR-005-1b",
    "stage_name": "일차침전지"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.3 혐기조 (`anaerobic`)

> **소스**: `wwtp_pipeline/stages/stage_03_anaerobic.py` — `Stage03Anaerobic`, `AnaerobicFeatureEngineer`

유입수, 1차침전지, 운전조건, 혐기조 현재 상태 28개 컬럼을 기반으로 1시간 뒤 혐기조 수질을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-2a |
| **min_data_length** | 4 (lag 3h + 1행) |
| **Lag 피처** | 24컬럼 × 3 lag (1h, 2h, 3h) = 72개 |
| **Rolling 피처** | 없음 |

**필수 입력 컬럼** (28개):

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| 유입수 | `유입_Q`, `유입_BOD`, `유입_COD`, `유입_TN`, `유입_TP`, `유입_SS`, `유입_TOC`, `유입_T`, `유입_pH` | 9 |
| 1차침전지 | `1차_SS_in`, `1차_SS_eff`, `1차_BOD_in`, `1차_BOD_eff`, `1차_S_NH_in`, `1차_S_NH_eff`, `1차_S_PO4_in`, `1차_S_PO4_eff`, `1차_Q_sludge` | 9 |
| 운전조건 | `운전_Q_ras`, `운전_ras_ratio`, `운전_Q_was`, `운전_SS_was` | 4 |
| 혐기조 현재 | `혐기_MLSS`, `혐기_DO`, `혐기_S_NO`, `혐기_S_NH`, `혐기_S_PO4`, `혐기_BOD` | 6 |

**출력 스키마** (`Stage03Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `혐기_S_PO4` | float | mg/L | 인산염 인 농도 (혐기 조건에서 방출 후 증가) |
| `혐기_MLSS` | float | mg/L | 혼합액 부유고형물 농도 |
| `혐기_BOD` | float | mg/L | 혐기조 BOD 농도 |

**Request 예시** (`POST /predict/anaerobic`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, 100000, 100000],
    "유입_BOD": [150.0, 150.0, 150.0, 150.0],
    "유입_COD": [250.0, 250.0, 250.0, 250.0],
    "유입_TN": [35.0, 35.0, 35.0, 35.0],
    "유입_TP": [5.0, 5.0, 5.0, 5.0],
    "유입_SS": [120.0, 120.0, 120.0, 120.0],
    "유입_TOC": [80.0, 80.0, 80.0, 80.0],
    "유입_T": [18.0, 18.0, 18.0, 18.0],
    "유입_pH": [7.2, 7.2, 7.2, 7.2],
    "1차_SS_in": [120.0, 120.0, 120.0, 120.0],
    "1차_SS_eff": [60.0, 60.0, 60.0, 60.0],
    "1차_BOD_in": [150.0, 150.0, 150.0, 150.0],
    "1차_BOD_eff": [105.0, 105.0, 105.0, 105.0],
    "1차_S_NH_in": [25.0, 25.0, 25.0, 25.0],
    "1차_S_NH_eff": [24.0, 24.0, 24.0, 24.0],
    "1차_S_PO4_in": [4.0, 4.0, 4.0, 4.0],
    "1차_S_PO4_eff": [3.8, 3.8, 3.8, 3.8],
    "1차_Q_sludge": [500.0, 500.0, 500.0, 500.0],
    "운전_Q_ras": [30000, 30000, 30000, 30000],
    "운전_ras_ratio": [30.0, 30.0, 30.0, 30.0],
    "운전_Q_was": [2000, 2000, 2000, 2000],
    "운전_SS_was": [8000, 8000, 8000, 8000],
    "혐기_MLSS": [2500, 2500, 2500, 2500],
    "혐기_DO": [0.2, 0.2, 0.2, 0.2],
    "혐기_S_NO": [0.5, 0.5, 0.5, 0.5],
    "혐기_S_NH": [25.0, 25.0, 25.0, 25.0],
    "혐기_S_PO4": [8.0, 8.0, 8.0, 8.0],
    "혐기_BOD": [100.0, 100.0, 100.0, 100.0]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "혐기_S_PO4": 8.5,
      "혐기_MLSS": 2520,
      "혐기_BOD": 98.5
    },
    "model_id": "MFR-005-2a",
    "stage_name": "혐기조"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.4 무산소조 (`anoxic`)

> **소스**: `wwtp_pipeline/stages/stage_04_anoxic.py` — `Stage04Anoxic`, `AnoxicFeatureEngineer`

유입수, 운전조건, 혐기조 및 무산소조 현재 상태 18개 컬럼을 기반으로 1시간 뒤 무산소조 수질을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-2b |
| **min_data_length** | 4 (lag 3h + 1행) |
| **Lag 피처** | 6컬럼(무산소) × 3 lag (1h, 2h, 3h) = 18개 |
| **Rolling 피처** | 없음 |

**필수 입력 컬럼** (18개):

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| 유입수 | `유입_Q`, `유입_T`, `유입_pH` | 3 |
| 운전조건 | `운전_Q_ir`, `운전_ir_ratio`, `운전_carbon_kg_d` | 3 |
| 혐기조 | `혐기_MLSS`, `혐기_DO`, `혐기_S_NO`, `혐기_S_NH`, `혐기_S_PO4`, `혐기_BOD` | 6 |
| 무산소조 현재 | `무산소_MLSS`, `무산소_DO`, `무산소_S_NO`, `무산소_S_NH`, `무산소_S_PO4`, `무산소_BOD` | 6 |

**출력 스키마** (`Stage04Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `무산소_S_NH` | float | mg/L | 암모니아성 질소 농도 |
| `무산소_S_NO` | float | mg/L | 질산성 질소 농도 (탈질 후 감소) |
| `무산소_MLSS` | float | mg/L | 혼합액 부유고형물 농도 |

**Request 예시** (`POST /predict/anoxic`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, 100000, 100000],
    "유입_T": [18.0, 18.0, 18.0, 18.0],
    "유입_pH": [7.2, 7.2, 7.2, 7.2],
    "운전_Q_ir": [200000, 200000, 200000, 200000],
    "운전_ir_ratio": [200.0, 200.0, 200.0, 200.0],
    "운전_carbon_kg_d": [1000.0, 1000.0, 1000.0, 1000.0],
    "혐기_MLSS": [2500, 2500, 2500, 2500],
    "혐기_DO": [0.2, 0.2, 0.2, 0.2],
    "혐기_S_NO": [0.5, 0.5, 0.5, 0.5],
    "혐기_S_NH": [25.0, 25.0, 25.0, 25.0],
    "혐기_S_PO4": [8.0, 8.0, 8.0, 8.0],
    "혐기_BOD": [100.0, 100.0, 100.0, 100.0],
    "무산소_MLSS": [2800, 2800, 2800, 2800],
    "무산소_DO": [0.3, 0.3, 0.3, 0.3],
    "무산소_S_NO": [5.0, 5.0, 5.0, 5.0],
    "무산소_S_NH": [20.0, 20.0, 20.0, 20.0],
    "무산소_S_PO4": [6.0, 6.0, 6.0, 6.0],
    "무산소_BOD": [80.0, 80.0, 80.0, 80.0]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "무산소_S_NH": 18.5,
      "무산소_S_NO": 3.2,
      "무산소_MLSS": 2850
    },
    "model_id": "MFR-005-2b",
    "stage_name": "무산소조"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.5 호기조 (`aerobic`)

> **소스**: `wwtp_pipeline/stages/stage_05_aerobic.py` — `Stage05Aerobic`, `AerobicFeatureEngineer`

유입수, 무산소조, 호기조 현재 상태, 송풍 및 운전조건 20개 컬럼을 기반으로 1시간 뒤 호기조 수질 5항목을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-2c |
| **min_data_length** | 4 (lag 3h + 1행) |
| **Lag 피처** | 5컬럼(호기) × 3 lag (1h, 2h, 3h) = 15개 |
| **Rolling 피처** | 없음 |

**필수 입력 컬럼** (20개):

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| 유입수 | `유입_Q`, `유입_T`, `유입_pH` | 3 |
| 무산소조 | `무산소_S_NO`, `무산소_S_NH`, `무산소_BOD`, `무산소_MLSS`, `무산소_DO` | 5 |
| 호기조 현재 | `호기_S_NO`, `호기_S_NH`, `호기_MLSS`, `호기_DO`, `호기_BOD`, `호기_S_PO4`, `호기_DO_setpoint` | 7 |
| 송풍 | `송풍_Q_air`, `송풍_n_running` | 2 |
| 운전조건 | `운전_Q_ir`, `운전_Q_ras`, `운전_Q_was` | 3 |

**출력 스키마** (`Stage05Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `호기_S_NO_next` | float | mg/L | 다음 시점 질산성 질소 (질산화 생성물) |
| `호기_S_NH_next` | float | mg/L | 다음 시점 암모니아성 질소 (질산화 후 감소) |
| `호기_MLSS_next` | float | mg/L | 다음 시점 MLSS |
| `호기_DO_next` | float | mg/L | 다음 시점 용존산소 |
| `호기_BOD_next` | float | mg/L | 다음 시점 BOD |

**Request 예시** (`POST /predict/aerobic`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, 100000, 100000],
    "유입_T": [18.0, 18.0, 18.0, 18.0],
    "유입_pH": [7.2, 7.2, 7.2, 7.2],
    "무산소_S_NO": [3.2, 3.2, 3.2, 3.2],
    "무산소_S_NH": [18.5, 18.5, 18.5, 18.5],
    "무산소_BOD": [80.0, 80.0, 80.0, 80.0],
    "무산소_MLSS": [2800, 2800, 2800, 2800],
    "무산소_DO": [0.3, 0.3, 0.3, 0.3],
    "호기_S_NO": [8.5, 8.5, 8.5, 8.5],
    "호기_S_NH": [1.2, 1.2, 1.2, 1.2],
    "호기_MLSS": [3500, 3500, 3500, 3500],
    "호기_DO": [2.1, 2.1, 2.1, 2.1],
    "호기_BOD": [8.0, 8.0, 8.0, 8.0],
    "호기_S_PO4": [0.5, 0.5, 0.5, 0.5],
    "호기_DO_setpoint": [2.0, 2.0, 2.0, 2.0],
    "송풍_Q_air": [5000.0, 5000.0, 5000.0, 5000.0],
    "송풍_n_running": [3, 3, 3, 3],
    "운전_Q_ir": [200000, 200000, 200000, 200000],
    "운전_Q_ras": [30000, 30000, 30000, 30000],
    "운전_Q_was": [2000, 2000, 2000, 2000]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "호기_S_NO_next": 9.1,
      "호기_S_NH_next": 0.9,
      "호기_MLSS_next": 3520,
      "호기_DO_next": 2.05,
      "호기_BOD_next": 7.2
    },
    "model_id": "MFR-005-2c",
    "stage_name": "호기조"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.6 이차침전지 (`secondary-clarifier`)

> **소스**: `wwtp_pipeline/stages/stage_06_secondary_clarifier.py` — `Stage06SecondaryClarifier`, `SecondaryClarifierFeatureEngineer`
> **알고리즘**: `WeightedMultiOutputXGB` (샘플 가중치 지원 Multi-Output XGBoost)

유입수, 호기조, 이차침전지 현재값, 운전조건 등을 기반으로 1시간 뒤 이차침전지 유출수 5항목을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-3a |
| **min_data_length** | 25 (rolling 24h + 1행) |
| **Lag 피처** | 10컬럼 × 4 lag (1h, 2h, 3h, 6h) = 40개 |
| **Rolling 피처** | 8컬럼 × 4 통계 (mean, std, max, min) × 24h = 32개 |

**필수 입력 컬럼**:

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| 유입수 | `유입_Q`, `유입_BOD`, `유입_SS`, `유입_TP`, `유입_TN`, `유입_TOC` | 6 |
| 이차침전지 현재 | `2차_BOD`, `2차_COD`, `2차_SS`, `2차_TN`, `2차_TP` | 5 |
| 호기조/도메인 | `호기_MLSS`, `호기_DO`, `호기_S_NO`, `호기_S_PO4` | 4 |
| 이차침전지 도메인 | `2차_DO` | 1 |
| 운전조건 | `운전_SRT`, `운전_HRT` | 2 |

> **참고**: 도메인 피처 소스 컬럼(`호기_MLSS`, `2차_SS`, `호기_DO`, `2차_DO`, `호기_S_NO`, `2차_TN`, `호기_S_PO4`, `2차_TP`, `운전_SRT`, `운전_HRT`)은 존재하는 경우에만 도메인 비율 피처(MLSS_SS_ratio 등)를 생성합니다.

**출력 스키마** (`Stage06Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `이차_BOD_next` | float | mg/L | 다음 시점 이차침전 유출 BOD |
| `이차_COD_next` | float | mg/L | 다음 시점 이차침전 유출 COD |
| `이차_SS_next` | float | mg/L | 다음 시점 이차침전 유출 SS |
| `이차_TN_next` | float | mg/L | 다음 시점 이차침전 유출 TN |
| `이차_TP_next` | float | mg/L | 다음 시점 이차침전 유출 TP |

**Request 예시** (`POST /predict/secondary-clarifier`):

```json
{
  "stage_input": {
    "유입_Q": [100000, "...(25개 시계열)"],
    "유입_BOD": [150.0, "..."],
    "유입_SS": [120.0, "..."],
    "유입_TP": [5.0, "..."],
    "유입_TN": [35.0, "..."],
    "유입_TOC": [80.0, "..."],
    "2차_BOD": [5.2, "..."],
    "2차_COD": [18.5, "..."],
    "2차_SS": [6.3, "..."],
    "2차_TN": [12.8, "..."],
    "2차_TP": [0.35, "..."],
    "호기_MLSS": [3500, "..."],
    "호기_DO": [2.1, "..."],
    "호기_S_NO": [8.5, "..."],
    "호기_S_PO4": [0.5, "..."],
    "2차_DO": [1.5, "..."],
    "운전_SRT": [10.0, "..."],
    "운전_HRT": [6.0, "..."]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "이차_BOD_next": 5.1,
      "이차_COD_next": 17.8,
      "이차_SS_next": 5.9,
      "이차_TN_next": 12.5,
      "이차_TP_next": 0.32
    },
    "model_id": "MFR-005-3a",
    "stage_name": "이차침전지"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.7 총인처리 (`tp-treatment`)

> **소스**: `wwtp_pipeline/stages/stage_07_total_phosphorus.py` — `Stage07TotalPhosphorus`, `TotalPhosphorusFeatureEngineer`

전 공정 데이터(유입수~이차침전지) 및 총인처리 현재 상태를 기반으로 1시간 뒤 총인처리 방류 TP/TN을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-4b |
| **min_data_length** | 4 (lag 3h + 1행) |
| **Lag 피처** | 24컬럼 × 3 lag (1h, 2h, 3h) = 72개 |
| **Rolling 피처** | 없음 |

**필수 입력 컬럼 (주요)**:

> 전체 입력은 `BASE_FEATURE_CANDIDATES` (~74개) + `LAG_BASE_COLUMNS` (24개)의 합집합입니다. 아래는 핵심 필수 컬럼(`STRICT_REQUIRED_COLUMNS`)과 주요 그룹입니다.

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| **핵심 필수** | `유입_Q`, `유입_TP`, `유입_SS`, `총인처리_TP`, `총인처리_TN`, `총인처리_S_PO4`, `총인처리_coag_mg_L` | 7 |
| 유입수 | `유입_Q`, `유입_BOD`, `유입_COD`, `유입_TN`, `유입_TP`, `유입_SS`, `유입_TOC`, `유입_T`, `유입_pH` | 9 |
| 1차침전지 | `1차_SS_in/eff`, `1차_BOD_in/eff`, `1차_S_NH_in/eff`, `1차_S_PO4_in/eff`, `1차_Q_sludge` | 9 |
| 혐기조 | `혐기_MLSS`, `혐기_DO`, `혐기_S_NO`, `혐기_S_NH`, `혐기_S_PO4`, `혐기_BOD` | 6 |
| 무산소조 | `무산소_MLSS`, `무산소_DO`, `무산소_S_NO`, `무산소_S_NH`, `무산소_S_PO4`, `무산소_BOD` | 6 |
| 호기조 | `호기_MLSS`, `호기_DO`, `호기_S_NO`, `호기_S_NH`, `호기_S_PO4`, `호기_BOD`, `호기_DO_setpoint` | 7 |
| 송풍 | `송풍_Q_air`, `송풍_n_running` | 2 |
| 이차침전지 | `2차_DO`, `2차_S_NO`, `2차_S_NH`, `2차_S_PO4`, `2차_BOD`, `2차_COD`, `2차_TN`, `2차_TP`, `2차_SS`, `2차_SOR`, `2차_SLR`, `2차_TSS_blanket`, `2차_HRT` | 13 |
| 운전 | `운전_SRT`, `운전_HRT`, `운전_Q_ras`, `운전_ras_ratio`, `운전_Q_ir`, `운전_ir_ratio`, `운전_Q_was`, `운전_SS_was`, `운전_carbon_kg_d` | 9 |
| 총인처리 현재 | `총인처리_S_NO`, `총인처리_S_NH`, `총인처리_S_PO4`, `총인처리_BOD`, `총인처리_COD`, `총인처리_TN`, `총인처리_TP`, `총인처리_coag_mg_L`, `총인처리_coag_kg_d`, `총인처리_Q_backwash`, `총인처리_SS_in`, `총인처리_SS_eff`, `총인처리_SS_removal` | 13 |

**출력 스키마** (`Stage07Output`):

| 필드 | 타입 | 단위 | 설명 |
|------|------|------|------|
| `총인처리_TP_next` | float | mg/L | 다음 시점 총인처리 방류 TP |
| `총인처리_TN_next` | float | mg/L | 다음 시점 총인처리 방류 TN |

**Request 예시** (`POST /predict/tp-treatment`):

```json
{
  "stage_input": {
    "유입_Q": [100000, 100000, 100000, 100000],
    "유입_TP": [5.0, 5.0, 5.0, 5.0],
    "유입_SS": [120.0, 120.0, 120.0, 120.0],
    "총인처리_TP": [0.15, 0.15, 0.15, 0.15],
    "총인처리_TN": [12.0, 12.0, 12.0, 12.0],
    "총인처리_S_PO4": [0.08, 0.08, 0.08, 0.08],
    "총인처리_coag_mg_L": [30.0, 30.0, 30.0, 30.0],
    "...(나머지 컬럼 생략, 전체 ~74개 컬럼 필요)": "..."
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "총인처리_TP_next": 0.12,
      "총인처리_TN_next": 11.5
    },
    "model_id": "MFR-005-4b",
    "stage_name": "총인처리"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 2.8 방류수 (`effluent`)

> **소스**: `wwtp_pipeline/stages/stage_08_effluent.py` — `Stage08Effluent`, `EffluentFeatureEngineer`

전 공정 22개 핵심 컬럼의 시계열 데이터를 기반으로 1시간 뒤 방류수 수질 7항목을 예측합니다.

| 항목 | 값 |
|------|---|
| **모델 ID** | MFR-005-4a |
| **min_data_length** | 25 (lag 24h + 1행) |
| **Lag 피처** | 22컬럼 × 5 lag (1h, 2h, 3h, 6h, 24h) = 110개 |
| **Rolling 피처** | 7컬럼 × 2 윈도우 (6h, 24h) × 2 통계 (mean, std) = 28개 |

**필수 입력 컬럼** (22개):

| 그룹 | 컬럼명 | 개수 |
|------|--------|------|
| 유입수 | `유입_Q`, `유입_BOD`, `유입_COD`, `유입_TN`, `유입_TP`, `유입_SS` | 6 |
| 생물반응조 | `혐기_MLSS`, `무산소_MLSS`, `호기_MLSS`, `호기_DO` | 4 |
| 이차침전지 | `2차_TN`, `2차_TP`, `2차_BOD` | 3 |
| 총인처리 | `총인처리_TN`, `총인처리_TP` | 2 |
| 방류 현재값 | `방류_BOD`, `방류_TN`, `방류_TP`, `방류_SS`, `방류_NH4`, `방류_NO3`, `방류_COD` | 7 |

**출력 스키마** (`Stage08Output`):

| 필드 | 타입 | 단위 | 법적 기준 | 설명 |
|------|------|------|-----------|------|
| `방류_TN_next` | float | mg/L | ≤ 20 | 다음 시점 방류 총질소 |
| `방류_TP_next` | float | mg/L | ≤ 0.2 | 다음 시점 방류 총인 |
| `방류_BOD_next` | float | mg/L | ≤ 5 | 다음 시점 방류 BOD |
| `방류_SS_next` | float | mg/L | ≤ 10 | 다음 시점 방류 SS |
| `방류_NH4_next` | float | mg/L | - | 다음 시점 방류 암모니아 |
| `방류_NO3_next` | float | mg/L | - | 다음 시점 방류 질산염 |
| `방류_COD_next` | float | mg/L | ≤ 40 | 다음 시점 방류 COD |

**Request 예시** (`POST /predict/effluent`):

```json
{
  "stage_input": {
    "유입_Q": [100000, "...(25개 시계열)"],
    "유입_BOD": [150.0, "..."],
    "유입_COD": [250.0, "..."],
    "유입_TN": [35.0, "..."],
    "유입_TP": [5.0, "..."],
    "유입_SS": [120.0, "..."],
    "혐기_MLSS": [2500, "..."],
    "무산소_MLSS": [2800, "..."],
    "호기_MLSS": [3500, "..."],
    "호기_DO": [2.1, "..."],
    "2차_TN": [12.8, "..."],
    "2차_TP": [0.35, "..."],
    "2차_BOD": [5.2, "..."],
    "총인처리_TN": [11.5, "..."],
    "총인처리_TP": [0.12, "..."],
    "방류_BOD": [4.5, "..."],
    "방류_TN": [14.2, "..."],
    "방류_TP": [0.08, "..."],
    "방류_SS": [3.8, "..."],
    "방류_NH4": [0.5, "..."],
    "방류_NO3": [8.3, "..."],
    "방류_COD": [12.1, "..."]
  }
}
```

**Response 예시** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "predictions": {
      "방류_TN_next": 14.0,
      "방류_TP_next": 0.09,
      "방류_BOD_next": 4.3,
      "방류_SS_next": 3.5,
      "방류_NH4_next": 0.4,
      "방류_NO3_next": 8.1,
      "방류_COD_next": 11.8
    },
    "model_id": "MFR-005-4a",
    "stage_name": "방류수"
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}

---

## 3. 예측 이력 조회

### `GET /predict/history`

과거 예측 결과와 실측값을 비교 조회합니다.

> **구현 상태**: `NotImplementedError` (DB 연동 필요)

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `stage` | string | N | `null` | 공정 단계 (미지정 시 전체) |
| `start_date` | datetime | Y | - | 조회 시작일 (ISO 8601) |
| `end_date` | datetime | Y | - | 조회 종료일 (ISO 8601) |
| `interval` | string | N | `"1h"` | 집계 간격 (`1h` / `1d`) |

**Request 예시**:

```
GET /api/v1/predict/history?stage=effluent&start_date=2026-02-01T00:00:00Z&end_date=2026-02-19T23:59:59Z&interval=1d
```

**Response Body** (`200 OK`, `APIResponse`):

```json
{
  "success": true,
  "data": {
    "records": [
      {
        "timestamp": "2026-02-19T09:00:00+09:00",
        "stage": "effluent",
        "predicted": { "BOD": 4.5, "TN": 13.8 },
        "actual": { "BOD": 4.8, "TN": 14.1 },
        "error": { "BOD": 0.3, "TN": 0.3 }
      }
    ],
    "performance": {
      "BOD": { "R2": 0.89, "MAPE": 8.5 },
      "TN": { "R2": 0.87, "MAPE": 9.2 }
    }
  }
}
```

**에러 응답**:

| HTTP 코드 | 설명 |
|-----------|------|
| `400` | 잘못된 날짜 범위 또는 stage 값 |
| `422` | 필수 파라미터 누락 |

---

## 4. 방류수질 예측 시뮬레이션

### `POST /effluent/predict`

유입조건과 운전 제어값을 입력하면 방류수질을 예측합니다. 3일간 6시간 간격(12포인트) 예측 추이를 제공합니다.

> **기능명세서**: FUN-004-0200 (방류수 예측)
> **알고리즘**: XGBoost Multi-Output, R² ≥ 0.85
> **구현 상태**: `NotImplementedError`

**Request Body** (`EffluentPredictRequest`):

```json
{
  "influent": {
    "flow": 10000,
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "temperature": 18.5,
    "pH": 7.2
  },
  "operation": {
    "airflow": 5000,
    "IR_rate": 200,
    "carbon_dose": 100,
    "coagulant_dose": 50,
    "MLSS": 3500
  },
  "prediction_horizon": "72h",
  "interval": "6h"
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `influent` | InfluentData | Y | - | 유입조건 (상세: 공통 스키마 참조) |
| `operation` | OperationData | Y | - | 운전 제어 설정값 (상세: 공통 스키마 참조) |
| `prediction_horizon` | string | N | `"72h"` | 예측 기간 |
| `interval` | string | N | `"6h"` | 예측 간격 |

**Response Body** (`200 OK`, `EffluentPredictResponse`):

```json
{
  "success": true,
  "data": {
    "applied_operation": {
      "airflow": { "value": 5000, "unit": "m³/hr" },
      "IR_rate": { "value": 200, "unit": "%" },
      "carbon_dose": { "value": 100, "unit": "L/day" },
      "coagulant_dose": { "value": 50, "unit": "L/day" },
      "MLSS": { "value": 3500, "unit": "mg/L" },
      "predicted_DO": { "value": 2.1, "unit": "mg/L" }
    },
    "current_prediction": {
      "BOD": { "value": 4.5, "unit": "mg/L", "limit": 5, "status": "양호" },
      "COD": { "value": 14.2, "unit": "mg/L", "limit": 40, "status": "양호" },
      "TN": { "value": 13.8, "unit": "mg/L", "limit": 20, "status": "양호" },
      "TP": { "value": 0.15, "unit": "mg/L", "limit": 0.2, "status": "양호" },
      "SS": { "value": 3.2, "unit": "mg/L", "limit": 10, "status": "양호" },
      "NH4N": { "value": 1.1, "unit": "mg/L" }
    },
    "forecast": [
      {
        "timestamp": "2026-02-19T16:00:00+09:00",
        "hours_ahead": 6,
        "BOD": 4.6, "COD": 14.5, "TN": 14.0, "TP": 0.16, "SS": 3.3
      },
      {
        "timestamp": "2026-02-19T22:00:00+09:00",
        "hours_ahead": 12,
        "BOD": 4.8, "COD": 14.8, "TN": 14.3, "TP": 0.17, "SS": 3.4
      }
    ],
    "warnings": []
  },
  "metadata": {
    "model_id": "MFR-005",
    "fun_id": "FUN-004-0200",
    "algorithm": "XGBoost Multi-Output",
    "r_squared": 0.87,
    "total_points": 12,
    "predicted_at": "2026-02-19T10:00:00+09:00"
  }
}
```

**에러 응답**:

| HTTP 코드 | 설명 |
|-----------|------|
| `422` | 입력 데이터 검증 실패 |
| `503` | 모델 로딩 실패 |

---

## 5. 방류수 실시간 감시

### `GET /effluent/current`

현재 방류수질 측정값과 법적 기준 대비 달성률을 조회합니다.

> **기능명세서**: FUN-001-0500 (방류수 감시)
> **구현 상태**: `NotImplementedError` (실시간 데이터 연동 필요)

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `include_trend` | bool | N | `false` | 30일 추이 데이터 포함 여부 |

**Response Body** (`200 OK`, `APIResponse`):

```json
{
  "success": true,
  "data": {
    "current": {
      "BOD": { "value": 4.8, "limit": 5, "achievement_rate": 96.0, "status": "pass" },
      "COD": { "value": 15.2, "limit": 40, "achievement_rate": 38.0, "status": "pass" },
      "SS": { "value": 3.5, "limit": 10, "achievement_rate": 35.0, "status": "pass" },
      "TN": { "value": 14.1, "limit": 20, "achievement_rate": 70.5, "status": "pass" },
      "TP": { "value": 0.18, "limit": 0.2, "achievement_rate": 90.0, "status": "pass" }
    },
    "alert_levels": {
      "TN": { "level": "normal", "thresholds": { "caution": 16, "warning": 18, "critical": 20 } }
    },
    "predictions": {
      "1h": { "BOD": 4.9, "TN": 14.3, "TP": 0.19 },
      "24h": { "BOD": 5.1, "TN": 14.8, "TP": 0.20 }
    }
  },
  "metadata": {
    "model_id": "MFR-005-4a/4b",
    "fun_id": "FUN-001-0500",
    "measured_at": "2026-02-19T10:00:00+09:00"
  }
}
```

**에러 응답**:

| HTTP 코드 | 설명 |
|-----------|------|
| `503` | 실시간 데이터 조회 불가 |
