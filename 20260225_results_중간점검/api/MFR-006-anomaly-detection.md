# MFR-006: 공정진단 및 이상탐지 모델

- **과업지시서**: MFR-006 — 전 공정 센서 데이터 기반 이상탐지, 이상유형 분류, 심각도 판정, 원인추정
- **데이터정의서 모델**: 공정진단/이상탐지 모델
- **기능명세서**: FUN-002-0100 (유입수 분석 AI 진단), FUN-002-0200 (생물반응조 진단)
- **Spec 모듈**: `004-anomaly-detection`

---

## 설계 기준 참조

> 이상탐지 판정의 기준이 되는 정상 운전 범위입니다.

### 설계/운전 기준

| 항목 | 정상 범위 | 단위 | 비고 | 출처 |
|------|-----------|------|------|------|
| DO (호기조) | 1.5 ~ 3.5 | mg/L | 기본 설정 2.0 | `design_capacity.md` |
| MLSS (호기조) | 3,000 ~ 4,000 | mg/L | 목표 3,500 | `design_capacity.md` |
| SRT | 5.4 ~ 14.6 | 일 | 목표 10일 | `design_capacity.md` |
| F/M비 | 0.1 ~ 0.5 | kgBOD/kgMLSS·d | | `design_capacity.md` |
| 유입유량 | 60,000 ~ 170,000 | m³/d | 설계 115,000 | `design_capacity.md` |
| RAS 비율 | 25 ~ 35 | % | 기본 30% | `design_capacity.md` |
| IR 비율 | 100 ~ 400 | % | 동적제어 | `design_capacity.md` |

### 유입수 기준

| 항목 | 정상 범위 | 단위 | 출처 |
|------|-----------|------|------|
| 유입 BOD | 100 ~ 300 | mg/L | `databook_columns.md` |
| 유입 TN | 20 ~ 60 | mg/L | `databook_columns.md` |
| 유입 TP | 2 ~ 8 | mg/L | `databook_columns.md` |

### 반응조 내부 파라미터

> **주의**: 아래 항목은 databook 및 설계 문서에 명시된 정상 범위가 없으며, A2O 공정의 일반적인 운전 원리에 근거하여 설정한 값입니다.

| 항목 | 정상 범위 | 단위 | 근거 |
|------|-----------|------|------|
| DO (혐기/무산소조) | 0 ~ 0.5 | mg/L | 혐기/무산소 조건 유지를 위해 DO가 낮아야 한다는 원리 |
| MLSS (혐기/무산소조) | 2,000 ~ 4,000 | mg/L | 호기조 목표 MLSS 3,500에서 유추 |
| S_NH (호기조) | 0 ~ 5 | mg/L | 질산화 완료 시 잔류 암모니아 수준 |
| S_NO (호기조) | 0 ~ 15 | mg/L | 질산화 후 생성되는 질산성질소 범위 |
| S_PO4 (호기조) | 0 ~ 2 | mg/L | PAO 인 흡수 후 잔류 인산염 수준 |

### 방류수 법적 기준 (I지역)

| 항목 | 허용 기준 | 단위 | 출처 |
|------|-----------|------|------|
| 방류 BOD | ≤ 5 | mg/L | `design_capacity.md`, `databook_measurable.md` |
| 방류 TN | ≤ 20 | mg/L | `design_capacity.md`, `databook_measurable.md` |
| 방류 TP | ≤ 0.2 | mg/L | `design_capacity.md`, `databook_measurable.md` |
| 방류 SS | ≤ 10 | mg/L | `design_capacity.md`, `databook_measurable.md` |

> **출처 경로**: `dataset/databook/` 하위 파일 기준
>
> **severity_scale 정의**:
> - 1~5 범위: 데이터 정의서 기준 (`docs/TOR/A2O_데이터_정의서_v1_0_수정.xlsx` → "AI 모델 정의" 시트)
> - 라벨링: 1(정보), 2(주의), 3(경고), 4(심각), 5(위험)
> - 그러나 `severity_scale`의 정보값이 `databook`에 없어 구현 불가능

---

## 1. 실시간 이상탐지

### `POST /anomaly/detect`

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

**Response Body — 정상 시** (`200 OK`):

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
    "detected_at": "2026-02-19T10:00:00+09:00"
  }
}
```

> **severity_scale 정의**:
> - 1~5 범위: 데이터 정의서 기준 (`docs/TOR/A2O_데이터_정의서_v1_0_수정.xlsx` → "AI 모델 정의" 시트)
> - 라벨링: 1(정보), 2(주의), 3(경고), 4(심각), 5(위험) 
> - 그러나 `severity_scale`의 정보값이 `databook`에 없어 구현 불가능

**Response Body — 이상 탐지 시** (`200 OK`):

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
      }
    ],
    "process_health": {
      "influent": { "status": "normal", "score": 0.95 },
      "anaerobic": { "status": "normal", "score": 0.92 },
      "anoxic": { "status": "normal", "score": 0.88 },
      "aerobic": { "status": "abnormal", "score": 0.35 },
      "effluent": { "status": "normal", "score": 0.94 }
    }
  },
  "metadata": {
    "detected_at": "2026-02-19T10:00:00+09:00"
  }
}
```

---

## 2. 이상 이력 조회

### `GET /anomaly/history`

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
