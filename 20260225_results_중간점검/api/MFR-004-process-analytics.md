# MFR-004: 공정 분석 모델

> **과업지시서**: MFR-004 — F/M비, SRT, HRT, 안정성지수 등 핵심 운영인자 분석
> **데이터정의서 모델**: MFR-004 (공정분석 모델)
> **기능명세서**: FUN-002-0100 (유입수 분석), FUN-002-0200 (생물반응조 분석·진단)
> **Spec 모듈**: `003-process-analytics`

**구현 대상 API (2개)**:
| # | Method | Endpoint | 설명 |
|---|--------|----------|------|
| 1 | POST | `/analytics/parameters` | 운영인자 계산 |
| 2 | POST | `/analytics/reactor-distribution` | 반응조별 부하 분배 |

> **참고**: `/analytics/statistics`와 `/analytics/correlation`은 과업지시서 MFR에 매핑되지 않아 [부록](appendix-func-spec-only.md)에서 정의합니다.

---

## 설계 기준 참조

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

---

## 1. 운영인자 자동 계산

### `POST /analytics/parameters`

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

## 2. 반응조별 부하 분배

### `POST /analytics/reactor-distribution`

전체 유입부하를 반응조별(혐기/무산소/호기)로 분배하여 산정합니다.

> **참고**: 공정 분석 시 각 반응조의 부하 배분 현황 파악을 위한 API
>
> **데이터 출처**:
> - `influent_flow`, `influent_quality`: `dataset/a2o/results_ml_measurable.csv`에서 제공 (유입_Q, 유입_BOD 등)
> - `reactor_volumes`: **설계 파라미터**로 CSV에 미포함. 설정 파일 또는 DB에서 관리 필요

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
| `influent_flow` | float | Y | m³/day | 유입유량 (CSV: 유입_Q) |
| `influent_quality` | object | Y | mg/L | 유입수질 (CSV: 유입_BOD, 유입_COD, 유입_TN, 유입_TP, 유입_SS) |
| `reactor_volumes` | object | Y | m³ | 반응조별 용량 (**설계값**: 설정 파일/DB 관리) |

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
