# MFR-003: 유입부하 예측 및 제거율 산정

- **과업지시서**: MFR-003 — 유입 오염물질의 부하량(kg/day)을 산정하고 각 수질항목별 제거율을 계산
- **데이터정의서 모델**: MFR-003 (유입부하/제거율 모델)
- **기능명세서**: FUN-001-0300 (유입부하 감시), FUN-000-0100 (통합 상황판)
- **Spec 모듈**: `002-load-removal-calc`

---

## 설계 기준 참조

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

> **출처**: `dataset/databook/design_capacity.md`, `dataset/databook/databook_facility.md`, `dataset/databook/databook_columns.md`

---

## 1. 부하량 산정

### `POST /load/calculate`

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

## 2. 제거율 산정

### `POST /removal/calculate`

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

