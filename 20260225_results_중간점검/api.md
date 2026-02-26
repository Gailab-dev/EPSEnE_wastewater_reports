# EPSEnE AI API 명세서

**문서번호**: EPSEnE-API-SPEC-v1.0
**작성일자**: 2026-02-19
**상태**: Draft
**Base URL**: `/api/v1`

---

## 1. 문서 개요

본 문서는 A2O 하수처리장 지능화 시스템(EPSEnE)의 AI API 명세서입니다.
과업지시서의 MFR-003 ~ MFR-008 요구사항과 기능명세서(A2O-FUNC-SPEC-v5.1)의 상세 기능명세를 기반으로 각 API의 Endpoint, Request, Response를 정의합니다.

### 1.1 ID 매핑 체계

| 과업지시서 ID | 과업지시서 명칭 | 데이터정의서 모델 ID | 기능명세서 FUN ID |
|-------------|---------------|-------------------|-----------------|
| MFR-003 | 유입부하 분석, 제거율 분석 모델 | MFR-003 (유입부하/제거율 모델) | FUN-001-0300, FUN-000-0100 |
| MFR-004 | 공정분석 모델 | MFR-004 (공정분석 모델) | FUN-002-0100, FUN-002-0200 |
| MFR-005 | 공정별 예측 모델 (공정시뮬레이션 포함) | MFR-005-1a ~ MFR-005-4b | FUN-001-0100 ~ FUN-001-0500, FUN-004-0200 |
| MFR-006 | 공정진단 및 이상탐지 모델 | 공정진단/이상탐지 모델 | FUN-002-0100, FUN-002-0200 |
| MFR-007 | 공정운영 의사결정지원 모델 | 의사결정지원 모델 | FUN-004-0100 |
| MFR-008 | 운전 및 운영 제어인자 도출 모델 | 최적제어 도출 모델 | FUN-003-0100, FUN-003-0200, FUN-003-0300 |

> **참고**: 기능명세서에서 MFR-008은 송풍량 최적화, MFR-009는 외부탄소원 최적화, MFR-010은 응집제 최적화로 참조됩니다. 본 문서에서는 과업지시서의 MFR-008(운전 및 운영 제어인자 도출)에 해당 기능들을 모두 포함합니다. 방류수질 예측(FUN-004-0200)은 과업지시서 MFR-005 하위에 포함됩니다.

### 1.2 공통 사항

#### 공통 Response 헤더

```
Content-Type: application/json
X-Request-Id: {uuid}
X-Model-Version: {model_version}
```

#### 공통 에러 Response

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "에러 설명",
    "details": {}
  },
  "timestamp": "2026-02-19T10:00:00+09:00"
}
```

#### 방류수질 법적 기준 (I지역)

| 항목 | 기준 (mg/L) |
|------|-----------|
| BOD | ≤ 5 |
| COD | ≤ 40 |
| SS | ≤ 10 |
| TN | ≤ 20 |
| TP | ≤ 0.2 |

---

## 2. MFR-003: 유입부하 분석, 제거율 분석 모델

> **과업지시서**: MFR-003 — 유입 오염물질의 부하량(kg/day)을 분석하고 각 수질항목별 제거율을 분석
> **데이터정의서 모델**: MFR-003 (유입부하/제거율 모델)
> **기능명세서**: FUN-001-0300 (유입부하 감시), FUN-000-0100 (통합 상황판)
> **Spec 모듈**: `002-load-removal-calc`

### 설계 기준 참조

> 부하량 산정의 기준이 되는 설계용량입니다.

| 항목 | 설계값 | 단위 | 설계부하량 (kg/d) |
|------|--------|------|-------------------|
| 설계유입유량 (Q) | 115,000 | m³/d | - |
| 설계 BOD | 200 | mg/L | 23,000 |
| 설계 COD | 320 | mg/L | 36,800 |
| 설계 TN | 40 | mg/L | 4,600 |
| 설계 TP | 4 | mg/L | 460 |
| 설계 SS | 200 | mg/L | 23,000 |

> **유입수질 입력 범위** (databook 기준):

| 항목 | 범위 | 단위 |
|------|------|------|
| 유입유량 (Q) | 60,000 ~ 170,000 | m³/d |
| BOD | 100 ~ 300 | mg/L |
| COD | 160 ~ 480 | mg/L |
| TN | 20 ~ 60 | mg/L |
| TP | 2 ~ 8 | mg/L |
| SS | 150 ~ 400 | mg/L |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

### 2.1 부하량 산정

#### `POST /load/calculate`

유입유량과 수질 농도를 기반으로 부하량(kg/day)을 산정합니다.

**Request Body**:

```json
{
  "influent_flow": 10000,
  "influent_quality": {
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "SS": 120.0
  }
}
```

| 필드 | 타입 | 필수 | 단위 | 설명 |
|------|------|------|------|------|
| `influent_flow` | float | Y | m³/day | 유입유량 |
| `influent_quality.BOD` | float | Y | mg/L | 유입 BOD 농도 |
| `influent_quality.COD` | float | Y | mg/L | 유입 COD 농도 |
| `influent_quality.TN` | float | Y | mg/L | 유입 TN 농도 |
| `influent_quality.TP` | float | Y | mg/L | 유입 TP 농도 |
| `influent_quality.SS` | float | Y | mg/L | 유입 SS 농도 |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "loads": {
      "BOD": 1500.0,
      "COD": 2500.0,
      "TN": 350.0,
      "TP": 50.0,
      "SS": 1200.0
    }
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **산출 공식**: `부하량(kg/day) = 유입유량(m³/day) × 농도(mg/L) / 1000`

---

### 2.2 제거율 산정

#### `POST /removal/calculate`

유입수와 방류수 농도 데이터를 기반으로 수질 항목별 제거율(%)을 계산합니다.

**Request Body**:

```json
{
  "influent_quality": {
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "SS": 120.0
  },
  "effluent_quality": {
    "BOD": 5.0,
    "COD": 15.0,
    "TN": 12.0,
    "TP": 0.3,
    "SS": 4.0
  }
}
```

| 필드 | 타입 | 필수 | 단위 | 설명 |
|------|------|------|------|------|
| `influent_quality` | object | Y | mg/L | 유입수 농도 |
| `effluent_quality` | object | Y | mg/L | 방류수 농도 |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "removal_rates": {
      "BOD": { "value": 96.67, "unit": "%" },
      "COD": { "value": 94.00, "unit": "%" },
      "TN": { "value": 65.71, "unit": "%" },
      "TP": { "value": 94.00, "unit": "%" },
      "SS": { "value": 96.67, "unit": "%" }
    }
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00",
  }
}
```
> **산출 공식**: removal_rate = (influent - effluent) / influent × 100

---

---

## 3. MFR-004: 공정분석 모델

> **과업지시서**: MFR-004 — F/M비, SRT, HRT, 안정성지수 등 핵심 운영인자 분석
> **데이터정의서 모델**: MFR-004 (공정분석 모델)
> **기능명세서**: FUN-002-0100 (유입수 분석), FUN-002-0200 (생물반응조 분석·진단)
> **Spec 모듈**: `003-process-analytics`

### 설계 기준 참조

> 공정 분석의 기준이 되는 반응조 설계용량 및 운전 파라미터 범위입니다.

**반응조 설계용량**:

| 반응조 | 체적 (m³) | HRT (115,000 m³/d 기준) | 비율 |
|--------|-----------|-------------------------|------|
| 혐기조 | 8,000 | 1.7시간 | 13% |
| 무산소조 | 16,500 | 3.4시간 | 27% |
| 호기조 | 36,806 | 7.7시간 | 60% |
| **합계** | **61,306** | **12.8시간** | 100% |

**운전 파라미터 범위**:

| 항목 | 설계값 | 범위 | 단위 |
|------|--------|------|------|
| MLSS 목표 | 3,500 | - | mg/L |
| DO 설정 | 2.0 | 1.5 ~ 3.5 | mg/L |
| SRT 목표 | 10.0 | 5.4 ~ 14.6 | 일 |
| F/M비 | - | 0.1 ~ 0.5 | kgBOD/kgMLSS·d |
| RAS 비율 | 30% | 25 ~ 35% | - |
| IR 비율 | 동적제어 | 100 ~ 400% | - |
| Q_was | 동적제어 | 500 ~ 8,000 | m³/d |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

### 3.1 운영인자 자동 계산

#### `POST /analytics/parameters`

MLSS, DO, 유량 등 측정 데이터를 기반으로 핵심 운영인자를 자동 계산합니다.

**Request Body**:

```json
{
  "measurements": {
    "MLSS_aerobic": 3500,
    "SVI": 120,
    "influent_flow": 115000,
    "influent_BOD": 150,
    "aerobic_volume": 36806,
    "waste_sludge": 4000,
    "DO_aerobic": 2.1
  }
}
```

| 필드 | 타입 | 필수 | 단위 | 설명 |
|------|------|------|------|------|
| `MLSS_aerobic` | float | Y | mg/L | 호기조 MLSS |
| `SVI` | float | Y | mL/g | 슬러지 용량 지수 |
| `influent_flow` | float | Y | m³/day | 유입유량 |
| `influent_BOD` | float | Y | mg/L | 유입 BOD |
| `aerobic_volume` | float | Y | m³ | 호기조 용량 |
| `waste_sludge` | float | Y | m³/day | 잉여슬러지량 |
| `DO_aerobic` | float | N | mg/L | 호기조 DO |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "parameters": {
      "FM_ratio": { "value": 0.134, "unit": "kgBOD/kgMLSS·d" },
      "SRT": { "value": 9.2, "unit": "day" },
      "HRT_actual": { "value": 7.7, "unit": "hr" },
      "OLR": { "value": 0.469, "unit": "kgBOD/m³·d" }
    }
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **산출 공식**:
> - `F/M비 = BOD부하(kg/d) / (MLSS × V)(kg)` — FUN-002-0200
> - `SRT = (MLSS × V) / WAS량 (day)` — FUN-002-0200
> - `HRT = V / Q (hr)` — FUN-002-0200
> - `OLR = BOD부하 / V (kg/m³·d)` — FUN-002-0200
>
> **정상 범위 판정 (`normal_range`, `status`)**: 본 API는 운영인자의 **계산값만 반환**합니다. 정상 범위(normal_range) 및 상태 판정(status: normal/warning/critical)은 **백엔드에서 설정 관리**되어야 하며, 다음 출처를 기반으로 설정합니다:
> - 하수처리장 **설계 사양서** (설계 F/M비, 설계 SRT 등)
> - **운영 지침** 및 표준 운영 절차
> - **현장 운영 경험**을 통한 최적 운영 범위
> - 처리장별 **설계 조건**에 따른 커스터마이징

---

> **참고**: `/analytics/statistics`와 `/analytics/correlation`은 과업지시서 MFR에 매핑되지 않아 [부록](#9-부록-기능명세서-전용-api-과업지시서-mfr-해당-없음)에서 정의합니다.

### 3.2 반응조별 부하 분배

#### `POST /analytics/reactor-distribution`

전체 유입부하를 반응조별(혐기/무산소/호기)로 분배하여 산정합니다.

> **기능명세서**: FUN-002-0200 (생물반응조 분석)

**Request Body**:

```json
{
  "influent_flow": 115000,
  "influent_quality": {
    "BOD": 150.0,
    "COD": 250.0,
    "TN": 35.0,
    "TP": 5.0,
    "SS": 120.0
  },
  "reactor_volumes": {
    "anaerobic": 8000,
    "anoxic": 16500,
    "aerobic": 36806
  }
}
```

| 필드 | 타입 | 필수 | 단위 | 설명 |
|------|------|------|------|------|
| `influent_flow` | float | Y | m³/day | 유입유량 |
| `influent_quality` | object | Y | mg/L | 유입수질 |
| `reactor_volumes` | object | Y | m³ | 반응조별 용량 |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "distribution": {
      "anaerobic": {
        "BOD_load": 2251.0, "TN_load": 525.3, "TP_load": 75.0,
        "HRT": 1.7
      },
      "anoxic": {
        "BOD_load": 4641.9, "TN_load": 1083.0, "TP_load": 154.7,
        "HRT": 3.4
      },
      "aerobic": {
        "BOD_load": 10357.1, "TN_load": 2416.7, "TP_load": 345.3,
        "HRT": 7.7
      }
    },
    "total_HRT": 12.8
  },
  "metadata": {
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

## 4. MFR-005: 공정별 예측 모델 (공정시뮬레이션 포함)

> **과업지시서**: MFR-005 — 공정별 예측 모델 (공정시뮬레이션 포함). 유입수 → 1차침전지 → 혐기조 → 무산소조 → 호기조 → 2차침전지 → 총인처리 → 방류수 순차 예측
> **데이터정의서 모델**: MFR-005-1a ~ MFR-005-4b (8개 서브모델)
> **기능명세서**: FUN-001-0100 ~ FUN-001-0500 (공정 감시), FUN-000-0100 (통합 상황판)
> **Spec 모듈**: `001-wwtp-pipeline`
> **상세 명세**: [`docs/api/MFR-005-process-prediction.md`](api/MFR-005-process-prediction.md)

### 설계 기준 참조

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

### 4.1 전체 파이프라인 예측

#### `POST /predict/full-pipeline`

유입수 데이터를 입력받아 8단계 전체 공정의 수질을 순차 예측합니다.

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

| 필드 | 타입 | 필수 | 검증 | 단위 | 설명 |
|------|------|------|------|------|------|
| `influent.flow` | float | Y | `gt=0` | m³/day | 유입유량 |
| `influent.BOD` | float | Y | `ge=0` | mg/L | 유입 BOD |
| `influent.COD` | float | Y | `ge=0` | mg/L | 유입 COD |
| `influent.TN` | float | Y | `ge=0` | mg/L | 유입 TN |
| `influent.TP` | float | Y | `ge=0` | mg/L | 유입 TP |
| `influent.SS` | float | N | `ge=0` | mg/L | 유입 SS |
| `influent.TOC` | float | N | `ge=0` | mg/L | 유입 TOC |
| `influent.temperature` | float | N | - | ℃ | 수온 |
| `influent.pH` | float | N | `ge=0, le=14` | - | pH |
| `operation.MLSS` | float | N | `ge=0` | mg/L | 호기조 MLSS |
| `operation.DO_setpoint` | float | N | `ge=0` | mg/L | DO 설정값 |
| `operation.RAS_rate` | float | N | `ge=0` | % | 외부반송율 |
| `operation.IR_rate` | float | N | `ge=0` | % | 내부반송율 |
| `operation.carbon_dose` | float | N | `ge=0` | L/day | 외부탄소원 투입량 |
| `operation.coagulant_dose` | float | N | `ge=0` | L/day | 응집제 투입량 |
| `operation.WAS_volume` | float | N | `ge=0` | m³/day | 잉여슬러지 인발량 |
| `operation.airflow` | float | N | `ge=0` | m³/hr | 송풍량 |

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

---

### 4.2 개별 공정 예측

#### `POST /predict/{stage}`

개별 공정 단계의 수질을 예측합니다.

> **상세 명세**: [MFR-005-process-prediction.md](api/MFR-005-process-prediction.md) §2 참조

**Path Parameters** (`StageEnum`):

| stage | 모델 ID | 입력 컬럼 수 | min_data_length | 출력 항목 수 | 설명 |
|-------|--------|-------------|-----------------|-------------|------|
| `influent` | MFR-005-1a | 6 | 169 | 3 | 유입수 부하량 예측 |
| `primary-clarifier` | MFR-005-1b | 8 | 168 | 2 | 1차침전지 유출수 예측 |
| `anaerobic` | MFR-005-2a | 28 | 4 | 3 | 혐기조 수질 예측 |
| `anoxic` | MFR-005-2b | 18 | 4 | 3 | 무산소조 수질 예측 |
| `aerobic` | MFR-005-2c | 20 | 4 | 5 | 호기조 수질 예측 |
| `secondary-clarifier` | MFR-005-3a | 18+ | 25 | 5 | 이차침전지 유출수 예측 |
| `tp-treatment` | MFR-005-4b | ~74 | 4 | 2 | 총인처리 방류 예측 |
| `effluent` | MFR-005-4a | 22 | 25 | 7 | 방류수 수질 예측 |

**Request Body** (`StagePredictRequest`):

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `stage_input` | dict | Y | 공정별 입력 데이터. 값이 list인 경우 시계열로 처리 |
| `upstream_output` | dict | N | 상류 공정 출력 데이터 (자동 병합) |

**각 Stage별 출력 요약**:

| stage | 출력 필드 |
|-------|----------|
| `influent` | `BOD_부하량`, `TN_부하량`, `TP_부하량` (kg/일) |
| `primary-clarifier` | `일차침전_BOD_eff_next`, `일차침전_SS_eff_next` (mg/L) |
| `anaerobic` | `혐기_S_PO4`, `혐기_MLSS`, `혐기_BOD` (mg/L) |
| `anoxic` | `무산소_S_NH`, `무산소_S_NO`, `무산소_MLSS` (mg/L) |
| `aerobic` | `호기_S_NO_next`, `호기_S_NH_next`, `호기_MLSS_next`, `호기_DO_next`, `호기_BOD_next` (mg/L) |
| `secondary-clarifier` | `이차_BOD_next`, `이차_COD_next`, `이차_SS_next`, `이차_TN_next`, `이차_TP_next` (mg/L) |
| `tp-treatment` | `총인처리_TP_next`, `총인처리_TN_next` (mg/L) |
| `effluent` | `방류_TN_next`, `방류_TP_next`, `방류_BOD_next`, `방류_SS_next`, `방류_NH4_next`, `방류_NO3_next`, `방류_COD_next` (mg/L) |

**Response 예시** (`200 OK`, `StagePredictResponse`):

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

### 4.3 예측 이력 조회

#### `GET /predict/history`

과거 예측 결과와 실측값을 비교 조회합니다.

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `stage` | string | N | `null` | 공정 단계 (미지정 시 전체) |
| `start_date` | datetime | Y | - | 조회 시작일 (ISO 8601) |
| `end_date` | datetime | Y | - | 조회 종료일 |
| `interval` | string | N | `"1h"` | `1h` / `1d` |

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

---

## 5. MFR-005 (확장): 방류수질 예측 시뮬레이션

> **과업지시서**: MFR-005 — 공정별 예측 모델 (공정시뮬레이션 포함). 방류수질 예측은 MFR-005의 최종 단계
> **데이터정의서 모델**: MFR-005-4a/4b (방류수 예측 모델 + 총인처리 예측 모델)
> **기능명세서**: FUN-004-0200 (방류수 예측), FUN-001-0500 (방류수 감시)
> **Spec 모듈**: `001-wwtp-pipeline` (방류수 부분)
> **알고리즘**: XGBoost Multi-Output, R² ≥ 0.85

### 5.1 방류수질 예측 (시뮬레이션)

#### `POST /effluent/predict`

유입조건과 운전 제어값을 입력하면 방류수질을 예측합니다. 3일간 6시간 간격(12포인트) 예측 추이를 제공합니다.

> **기능명세서**: FUN-004-0200

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
| `influent` | InfluentData | Y | - | 유입조건 (상세: Section 4 참조) |
| `operation` | OperationData | Y | - | 운전 제어 설정값 (상세: Section 4 참조) |
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

---

### 5.2 방류수 실시간 감시

#### `GET /effluent/current`

현재 방류수질 측정값과 법적 기준 대비 달성률을 조회합니다.

> **기능명세서**: FUN-001-0500 (방류수 감시)

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

---

## 6. MFR-006: 공정진단 및 이상탐지 모델

> **과업지시서**: MFR-006 — 전 공정 센서 데이터 기반 이상탐지, 이상유형 분류, 심각도 판정, 원인추정
> **데이터정의서 모델**: 공정진단/이상탐지 모델
> **기능명세서**: FUN-002-0100 (유입수 분석 AI 진단), FUN-002-0200 (생물반응조 진단)
> **Spec 모듈**: `004-anomaly-detection`

### 설계 기준 참조

> 이상탐지 판정의 기준이 되는 정상 운전 범위입니다.

| 항목 | 정상 범위 | 단위 | 비고 |
|------|-----------|------|------|
| DO (호기조) | 1.5 ~ 3.5 | mg/L | 기본 설정 2.0 |
| MLSS (호기조) | 3,000 ~ 4,000 | mg/L | 목표 3,500 |
| SRT | 5.4 ~ 14.6 | 일 | 목표 10일 |
| F/M비 | 0.1 ~ 0.5 | kgBOD/kgMLSS·d | |
| 유입유량 | 60,000 ~ 170,000 | m³/d | 설계 115,000 |
| RAS 비율 | 25 ~ 35 | % | 기본 30% |
| IR 비율 | 100 ~ 400 | % | 동적제어 |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

### 6.1 실시간 이상탐지

#### `POST /anomaly/detect`

전 공정 센서값을 기반으로 이상 여부를 실시간 탐지합니다.

**Request Body**:

```json
{
  "sensors": {
    "influent": { "flow": 10000, "BOD": 150, "TN": 35, "TP": 5, "pH": 7.2, "temperature": 18.5 },
    "anaerobic": { "MLSS": 2800, "DO": 0.2, "S_NO": 0.1, "S_NH": 15.0, "S_PO4": 8.5, "BOD": 120 },
    "anoxic": { "MLSS": 2900, "DO": 0.3, "S_NO": 2.1, "S_NH": 12.0, "S_PO4": 6.2, "BOD": 95 },
    "aerobic": { "MLSS": 3500, "DO": 2.1, "S_NH": 1.2, "S_NO": 8.5, "S_PO4": 0.5, "BOD": 8.0 },
    "effluent": { "BOD": 4.8, "TN": 14.1, "TP": 0.18, "SS": 3.5 }
  },
  "operation": {
    "airflow": 5000,
    "RAS_rate": 50,
    "IR_rate": 200,
    "carbon_dose": 100
  }
}
```

> **데이터 출처**: `dataset/a2o/results_ml_measurable.csv`에서 제공되는 센서값만 사용
> - **포함**: MLSS, DO, S_NO, S_NH, S_PO4, BOD, pH, temperature, flow 등
> - **제외** (CSV 미포함): ORP, SVI, polymer_dose

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "is_abnormal": false,
    "anomaly_score": 0.12,
    "anomalies": [],
    "process_health": {
      "influent": { "status": "normal", "score": 0.95 },
      "anaerobic": { "status": "normal", "score": 0.92 },
      "anoxic": { "status": "normal", "score": 0.88 },
      "aerobic": { "status": "normal", "score": 0.91 },
      "effluent": { "status": "normal", "score": 0.94 }
    }
  },
  "metadata": {
    "severity_scale": "1(정보) ~ 5(위험)",
    "detected_at": "2026-02-19T10:00:00+09:00"
  }
}
```

**이상 탐지 시 Response 예시**:

```json
{
  "success": true,
  "data": {
    "is_abnormal": true,
    "anomaly_score": 0.78,
    "anomalies": [
      {
        "anomaly_type": "DO_DROP",
        "process": "aerobic",
        "severity": 3,
        "severity_label": "경고",
        "parameter": "DO",
        "current_value": 0.8,
        "normal_range": [1.0, 3.0],
        "probable_cause": "송풍기 이상 또는 급격한 유기물 부하 증가",
        "recommendations": [
          "송풍기 운전 상태 확인",
          "유입 BOD 부하 확인",
          "예비 송풍기 가동 검토"
        ]
      }
    ],
    "process_health": {
      "influent": { "status": "normal", "score": 0.95 },
      "aerobic": { "status": "abnormal", "score": 0.35 }
    }
  },
  "metadata": {
    "severity_scale": "1(정보) ~ 5(위험)",
    "detected_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

### 6.2 이상 이력 조회

#### `GET /anomaly/history`

과거 이상탐지 이력을 조회합니다.

**Query Parameters**:

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `start_date` | string | Y | 조회 시작일 |
| `end_date` | string | Y | 조회 종료일 |
| `process` | string | N | 공정 필터 |
| `severity_min` | int | N | 최소 심각도 (1~5) |
| `page` | int | N | 페이지 번호 (기본: 1) |
| `page_size` | int | N | 페이지 크기 (기본: 20) |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "total_count": 45,
    "records": [
      {
        "id": "ANM-20260219-001",
        "detected_at": "2026-02-19T08:30:00+09:00",
        "resolved_at": "2026-02-19T09:15:00+09:00",
        "anomaly_type": "DO_DROP",
        "process": "aerobic",
        "severity": 3,
        "probable_cause": "송풍기 #2 정지",
        "resolution": "예비 송풍기 가동"
      }
    ]
  }
}
```

---

## 7. MFR-007: 공정운영 의사결정지원 모델

> **과업지시서**: MFR-007 — 유입조건과 방류기준을 고려한 최적 운전제어값을 종합 도출하고, 운영자의 의사결정을 지원
> **데이터정의서 모델**: 의사결정지원 모델
> **기능명세서**: FUN-004-0100 (최적 운전제어값)
> **Spec 모듈**: `006-control-optimization`

### 설계 기준 참조

> 본 API의 유입조건 및 운전제어값은 다음 설계 기준을 참조합니다.

| 항목 | 설계값 | 단위 | 비고 |
|------|--------|------|------|
| 설계유입유량 | 115,000 | m³/d | 시간최대 170,000 |
| 설계 BOD | 200 | mg/L | |
| 설계 TN | 40 | mg/L | |
| 설계 TP | 4 | mg/L | |
| 송풍기 용량 | 6대 × 300 m³/min | m³/min | 총 1,800 m³/min |
| DO 범위 | 1.5 ~ 3.5 | mg/L | 기본 설정 2.0 |
| IR 범위 | 100 ~ 400 | % | 동적제어 |
| 외부탄소원 | 메탄올 0 ~ 20,000 | kg/d | C/N 피드포워드 제어 |
| 응집제 (PAC) | 30 ~ 100 | mg/L | 동적제어 |
| MLSS 목표 | 3,500 | mg/L | |

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`

### 7.1 통합 최적제어값 도출

#### `POST /optimization/optimal-control`

유입조건과 목표 방류수질을 입력하면 AI가 최적 운전제어값을 추천합니다.

> **기능명세서**: FUN-004-0100 (최적 운전제어값)
> **AI API 시트 요청**: 각 항목 하단 설명 텍스트 필요, Bar 차트 데이터

**Request Body**:

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
  "target_effluent": {
    "BOD": 10.0,
    "COD": 40.0,
    "TN": 20.0,
    "TP": 2.0
  }
}
```

| 필드 | 타입 | 필수 | 단위 | 설명 |
|------|------|------|------|------|
| `influent` | object | Y | - | 유입조건 (Q, BOD, COD, TN, TP, T, pH) |
| `target_effluent` | object | Y | mg/L | 방류수질 목표값 |

**Response Body** (`200 OK`):

```json
{
  "success": true,
  "data": {
    "optimal_controls": {
      "airflow": {
        "value": 4800, "unit": "m³/hr",
        "description": "현재 대비 4% 감소 — DO 2.0 유지 가능, 에너지 절감"
      },
      "DO_setpoint": {
        "value": 2.0, "unit": "mg/L",
        "description": "질산화 효율 유지, NH₄-N 1.5 이하 목표"
      },
      "IR_rate": {
        "value": 200, "unit": "%",
        "description": "탈질 효율 최적화 — 내부반송 NO₃-N 8.5 mg/L 기준"
      },
      "carbon_dose": {
        "value": 110, "unit": "L/day",
        "description": "C/N비 보정 — TN 방류기준 준수를 위한 소폭 증가"
      },
      "polymer_dose": {
        "value": 50, "unit": "L/day",
        "description": "현재 유지 — TP 방류기준 충분히 충족"
      },
      "MLSS_target": {
        "value": 3500, "unit": "mg/L",
        "description": "현재 유지 — F/M비 0.11 적정 범위"
      }
    },
    "predicted_effluent": {
      "BOD": { "value": 4.5, "limit": 10, "status": "기준_만족" },
      "COD": { "value": 14.2, "limit": 40, "status": "기준_만족" },
      "TN": { "value": 13.8, "limit": 20, "status": "기준_만족" },
      "TP": { "value": 0.15, "limit": 2, "status": "기준_만족" },
      "SS": { "value": 3.2, "limit": 10, "status": "기준_만족" },
      "NH4N": { "value": 1.1 }
    },
    "feasibility": {
      "achievable": true,
      "message": "목표 수질 달성 가능"
    },
    "chart_data": {
      "comparison": {
        "labels": ["BOD", "COD", "TN", "TP"],
        "influent": [150.0, 250.0, 35.0, 5.0],
        "predicted_effluent": [4.5, 14.2, 13.8, 0.15],
        "discharge_limits": [5, 40, 20, 0.2]
      }
    }
  },
  "metadata": {
    "model_id": "MFR-007",
    "fun_id": "FUN-004-0100",
    "optimization_method": "reverse_optimization",
    "calculated_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

---

## 8. MFR-008: 운전 및 운영 제어인자 도출 모델

> **과업지시서**: MFR-008 — 개별 제어인자(송풍량, 외부탄소원, 응집제)에 대한 최적값 도출
> **데이터정의서 모델**: 최적제어 도출 모델
> **기능명세서**: FUN-003-0100 (송풍량 최적화), FUN-003-0200 (외부탄소원 최적화), FUN-003-0300 (응집제 최적화)
> **Spec 모듈**: `006-control-optimization`

> **참고**: 기능명세서에서 MFR-008(송풍량), MFR-009(외부탄소원), MFR-010(응집제)으로 분리 참조되는 기능들을 과업지시서 MFR-008 하에 통합 정리합니다.

### 설계 기준 참조

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

### 8.1 송풍량 최적화

#### `POST /optimization/aeration`

AI가 추천하는 최적 송풍량과 DO 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0100 (송풍량 최적화)
> **기능명세서 AI 모델 참조**: MFR-008

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

---

### 8.2 외부탄소원 최적화

#### `POST /optimization/carbon`

AI가 추천하는 최적 외부탄소원 투입량과 TN 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0200 (외부탄소원 최적화)
> **기능명세서 AI 모델 참조**: MFR-009

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

---

### 8.3 응집제 최적화

#### `POST /optimization/coagulant`

AI가 추천하는 최적 응집제 투입량과 TP 변화 예측을 제공합니다.

> **기능명세서**: FUN-003-0300 (응집제 최적화)
> **기능명세서 AI 모델 참조**: MFR-010

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
> - **법적 기준**: TP 방류기준 ≤0.2 mg/L (I지역 기준, 고정값)

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

---

## 9. 부록: 기능명세서 전용 API (과업지시서 MFR 해당 없음)

아래 API들은 기능명세서(A2O-FUNC-SPEC-v5.1)에 정의되어 있으나, 과업지시서 MFR-003~MFR-008에 직접 매핑되지 않는 기능입니다.

### 9.1 통계 분석

#### `POST /analytics/statistics`

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

### 9.2 상관관계 분석

#### `POST /analytics/correlation`

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

### 9.3 대시보드 최적화 권고 조회

#### `GET /dashboard/optimization-summary`

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

### 9.4 생물반응조 AI 자동 제어 상태 조회

#### `GET /bioreactor/ai-auto-status`

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

### 9.5 유입수 종합 분석 (군집 판정 + AI 진단)

#### `POST /analytics/influent-analysis`

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

### 9.6 생물반응조 종합 분석·진단

#### `POST /analytics/bioreactor-analysis`

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

### 9.7 원천데이터 조회

#### `GET /data/raw`

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

### 9.8 기능별 데이터 조회

#### `GET /data/function/{function_name}`

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

### 9.9 다중 추이 분석

#### `POST /data/multi-trend`

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

---

## 10. 전체 API 엔드포인트 요약

| # | Method | Endpoint | MFR | FUN ID | 설명 |
|---|--------|----------|-----|--------|------|
| 1 | POST | `/load/calculate` | MFR-003 | FUN-001-0300 | 부하량 산정 |
| 2 | POST | `/removal/calculate` | MFR-003 | FUN-001-0300 | 제거율 산정 |
| 3 | POST | `/analytics/parameters` | MFR-004 | FUN-002-0200 | 운영인자 계산 |
| 4 | POST | `/analytics/reactor-distribution` | MFR-004 | FUN-002-0200 | 반응조별 부하 분배 |
| 5 | POST | `/analytics/statistics` | — | FUN-005-0500 | 통계 분석 |
| 6 | POST | `/analytics/correlation` | — | FUN-005-0400 | 상관관계 분석 |
| 7 | POST | `/predict/full-pipeline` | MFR-005 | FUN-001-0100~0500 | 전체 파이프라인 예측 |
| 8 | POST | `/predict/{stage}` | MFR-005 | FUN-001-0100~0500 | 개별 공정 예측 |
| 9 | GET | `/predict/history` | MFR-005 | — | 예측 이력 조회 |
| 10 | POST | `/effluent/predict` | MFR-005 | FUN-004-0200 | 방류수질 예측 시뮬레이션 |
| 11 | GET | `/effluent/current` | MFR-005 | FUN-001-0500 | 방류수 실시간 감시 |
| 12 | POST | `/anomaly/detect` | MFR-006 | FUN-002-0100/0200 | 실시간 이상탐지 |
| 13 | GET | `/anomaly/history` | MFR-006 | — | 이상 이력 조회 |
| 14 | POST | `/optimization/optimal-control` | MFR-007 | FUN-004-0100 | 통합 최적제어값 도출 (의사결정지원) |
| 15 | POST | `/optimization/aeration` | MFR-008 | FUN-003-0100 | 송풍량 최적화 |
| 16 | POST | `/optimization/carbon` | MFR-008 | FUN-003-0200 | 외부탄소원 최적화 |
| 17 | POST | `/optimization/coagulant` | MFR-008 | FUN-003-0300 | 응집제 최적화 |
| 18 | GET | `/dashboard/optimization-summary` | — | FUN-000-0100 | 대시보드 최적화 요약 |
| 19 | GET | `/bioreactor/ai-auto-status` | — | FUN-001-0400 | AI 자동 제어 상태 |
| 20 | POST | `/analytics/influent-analysis` | — | FUN-002-0100 | 유입수 종합 분석 |
| 21 | POST | `/analytics/bioreactor-analysis` | — | FUN-002-0200 | 생물반응조 분석·진단 |
| 22 | GET | `/data/raw` | — | FUN-005-0100 | 원천데이터 조회 |
| 23 | GET | `/data/function/{function_name}` | — | FUN-005-0200 | 기능별 데이터 조회 |
| 24 | POST | `/data/multi-trend` | — | FUN-005-0300 | 다중 추이 분석 |

> **총 24개 엔드포인트** (MFR 매핑: 15개, 기능명세서 전용: 9개)

---

## 11. 참고 문서

| 문서 | 파일명 | 설명 |
|------|--------|------|
| 과업지시서 | `과업지시서_하수처리시설 공정지능화 솔루션 개발_2025_09_최종본.pdf` | MFR-001~MFR-008 요구사항 정의 |
| 기능명세서 | `A2O_기능목록_및_기능명세서_v5_2_수정.xlsx` | 22개 기능의 상세 명세 (FUN-000~FUN-006) |
| 데이터정의서 | `A2O_데이터_정의서_v1_0_수정.xlsx` | 원천데이터, AI 모델 I/O, 피처 스토어 |
| Pipeline Spec | `specs/001-wwtp-pipeline/spec.md` | 8단계 파이프라인 상세 설계 |
| Load/Removal Spec | `specs/002-load-removal-calc/spec.md` | 부하량/제거율 계산 설계 |
| Analytics Spec | `specs/003-process-analytics/spec.md` | 공정분석 설계 |
| Anomaly Spec | `specs/004-anomaly-detection/spec.md` | 이상탐지 설계 |
| Optimization Spec | `specs/006-control-optimization/spec.md` | 최적제어 설계 |
| 통합 API Spec | `specs/007-wwtp-ai-api/spec.md` | 통합 API 서비스 설계 |
