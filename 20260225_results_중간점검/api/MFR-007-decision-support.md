# MFR-007: 공정운영 의사결정지원 모델

- **과업지시서**: MFR-007 — 공정운영 의사결정지원 모델. 유입조건과 방류기준을 고려하여 최적 운전제어값을 종합 도출하고, 운영자의 의사결정을 지원
- **데이터정의서 모델**: 의사결정지원 모델
- **기능명세서**: FUN-004-0100 (최적 운전제어값)
- **Spec 모듈**: `006-control-optimization`

---

## 설계 기준 참조

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

---

## 1. 통합 최적제어값 도출

### `POST /optimization/optimal-control`

유입조건과 목표 방류수질을 입력하면 AI가 최적 운전제어값을 종합 추천합니다.

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

| 필드 | 타입 | 설명 |
|------|------|------|
| `optimal_controls.*` | object | 운전제어 최적값 6개. 각 항목에 `value`, `unit`, `description` 포함 |
| `predicted_effluent.*` | object | 예측 방류수질 6개. `limit`은 `target_effluent` 입력값 기준 |
| `predicted_effluent.*.status` | string | `"기준_만족"` / `"기준_초과"` (기능명세서 FUN-004-0100 용어) |
| `feasibility` | object | 목표수질 달성 가능 여부 판정 (기능명세서 제약조건) |
| `chart_data.comparison` | object | Bar 차트 + 점선 Annotation 데이터 |
| `chart_data.comparison.discharge_limits` | array | 방류 기준선 — 기능명세서 정의값 `[BOD 5, COD 40, TN 20, TP 0.2]` |

> **참고**: `predicted_effluent.limit`은 사용자가 입력한 `target_effluent` 값이고, `chart_data.discharge_limits`는 기능명세서 FUN-004-0100에서 정의한 차트 Annotation 기준선입니다. 두 값은 서로 다를 수 있습니다.

---

## Issues: 과업지시서 — 기능명세서 미정의 항목

아래 항목은 과업지시서 MFR-007에서 요구하지만, 기능명세서 FUN-004-0100에는 입출력이 정의되지 않아 현재 API에 미반영된 사항입니다.

### Issue 1: 이상탐지 모델 연동

- **과업지시서**: "공정분석과 공정진단 및 이상탐지 모델과 연동하여 비정상 운전인자를 감지, 분석할 수 있어야 함"
- **기능명세서**: FUN-004-0100에 이상탐지(MFR-006) 연동 입출력 정의 없음
- **현재 상태**: 미반영 — 이상탐지는 MFR-006(`/anomaly/detect`)에서 독립 수행
- **검토 필요**: 이상탐지 결과를 Request에 포함하거나, 서버 내부에서 MFR-006 결과를 자동 참조할지 결정 필요

### Issue 2: 처리공정별 운영인자 구분 및 비정상 해소 판정

- **과업지시서**: "처리공정별로 적정 운영인자를 제시하고, 제시된 적정 운영인자를 적용할 경우의 비정상 상태 해소 여부를 확인할 수 있어야 함"
- **기능명세서**: 운전제어 최적값은 통합 6개 항목으로 정의 (송풍량, DO, 내부반송율, 외부탄소원, 응집제, MLSS). 공정별 구분 없음. 비정상 해소 판정 필드 미정의
- **현재 상태**: 기능명세서 기준 통합 구조 유지. `feasibility`는 목표 달성 가능 여부만 판정

### Issue 3: 의사결정 이력 저장

- **과업지시서**: "DB 서버에 이력관리, 향후 재학습시 데이터로 활용가능하도록 의사결정지원정보가 실시간 저장될 수 있어야 함"
- **기능명세서**: 이력 ID, 저장 확인 관련 출력 필드 미정의
- **현재 상태**: 미반영 — 서버 내부 저장은 구현 단계에서 처리하되, Response에 `decision_id` 등 이력 식별자 추가 여부 검토 필요
