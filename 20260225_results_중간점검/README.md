# EPSEnE AI 시스템 중간점검 보고서

**작성일**: 2026-02-26
**프로젝트**: A2O 하수처리장 지능화 시스템 (EPSEnE)

---

## 1. 모델 구현 상태

> 상세: [model_performance_report.md](./model_performance_report.md) | [wwtp_pipeline.md](./wwtp_pipeline.md)

### 1.1 공정 예측 모델 (Stage 01~08)

8단계 공정별 XGBoost Multi-Output 모델 학습 및 pkl 저장 완료.

| Stage | 공정 | 모델 ID | 학습 | 성능 미달 컬럼 | 비고 |
|-------|------|---------|:----:|:--------------:|------|
| 01 | 유입수 | MFR-005-1a | ✅ | 0 | R² ≥ 0.999 전 항목 |
| 02 | 일차침전지 | MFR-005-1b | ✅ | 2 | BOD R²=0.33, SS R²=0.24 |
| 03 | 혐기조 | MFR-005-2a | ✅ | 2 | S_PO4 R²=0.85, BOD R²=0.81 |
| 04 | 무산소조 | MFR-005-2b | ✅ | 1 | S_NO R²=0.73, MAPE=20.3% |
| 05 | 호기조 | MFR-005-2c | ✅ | 2 | S_NH MAPE=16.6%, MLSS MAPE=15.2% |
| 06 | 이차침전지 | MFR-005-3a | ✅ | 1 | BOD R²=0.44 |
| 07 | 총인처리 | MFR-005-4b | ✅ | 0 | R² ≥ 0.97 전 항목 |
| 08 | 방류수 | MFR-005-4a | ✅ | 1 | BOD R²=0.62 |

> **성능 기준**: R² ≥ 0.85, MAPE ≤ 10%

### 1.2 모델 파일 위치

```
models/
├── 01_유입수/       model.pkl, scaler_X.pkl, scaler_y.pkl, feature_cols.pkl, target_cols.pkl
├── 02_일차침전지/
├── 03_혐기조/
├── 04_무산소조/
├── 05_호기조/       + meta.pkl
├── 06_이차침전지/
├── 07_총인처리/
├── 08_방류수/
└── 09_시뮬레이션/
```

---

## 2. API 명세서 정의 방식

> 상세: [api.md](./api.md)

### 2.1 문서 체계

```
docs/
├── api.md                          # 통합 API 명세서 (전체 MFR 포괄)
├── api/
│   ├── MFR-005-process-prediction.md  # MFR-005 상세 명세 (Stage별 입출력)
│   └── (추가 MFR별 상세 명세 예정)
└── MFR-005-API-test.md             # API 테스트 케이스 (JSON 예시)
```

- **통합 명세서** (`docs/api.md`): 전체 엔드포인트 15개의 Request/Response 정의
- **개별 상세 명세** (`docs/api/*.md`): Stage별 입력 컬럼, 출력 스키마, FeatureEngineer 사양
- **API 테스트** (`docs/MFR-005-API-test.md`): 각 Stage의 복사-붙여넣기 가능한 JSON 테스트 데이터

### 2.2 명세서 작성 현황

| MFR | 명세서 | 진행률 | 상태 |
|-----|--------|:------:|------|
| MFR-003 | 부하량/제거율 | **100%** | 완료 |
| MFR-004 | 공정분석 (운영인자) | **100%** | 완료 |
| MFR-005 | 공정별 예측 | **70%** | Stage별 상세 입출력 완료. Response 값 검증 필요 |
| MFR-006 | 이상탐지 | 작성 완료 | Response 값 검증 필요 |
| MFR-007 | 의사결정지원 | 초안 | 검토 필요 |
| MFR-008 | 최적제어 | 초안 | 검토 필요 |

### 2.3 메뉴-API 매핑

메뉴-파일-API 매핑 자료 제작 완료. 상세: [menu_mapping.md](./menu_mapping.md)

| 구분 | 수량 | 비고 |
|------|------|------|
| 사이드바 메뉴 | 19개 | |
| HTML 페이지 | 22개 | 메뉴 19 + 서브페이지 3 |
| MFR API 엔드포인트 | **15개** | 과업지시서 요구사항과 일치 |
| 부록 API 엔드포인트 | 9개 | 기능명세서(부록)에만 존재, 과업지시서에 없음 |
| 총 API 엔드포인트 | **24개** | |

- **과업지시서 vs 기능명세서 차이**: 
  - 기능명세서에서 요구하는 API 엔드포인트는 총 24개이나, 이 중 과업지시서에서 요구하는 것과 일치하는 것은 **15개 (MFR-003~008)**에 해당함
  - 나머지 9개는 기능명세서 부록(FUN-000~005)에서 정의된 대시보드 요약, 유입수 분석, 생물반응조 분석, 원천데이터 조회, 기능별 데이터, 다중추이, 통계, 상관분석, AI자동제어 상태 API로, 과업지시서의 MFR 요구사항과 상이한 부분이 있어 개발범위 협의가 필요함

---

## 3. API 구현 상황

> 상세: [api.md](./api.md)

### 3.1 엔드포인트별 구현 현황

| # | Method | Endpoint | MFR | 구현 상태 | 비고 |
|---|--------|----------|-----|----------|------|
| 1 | POST | `/load/calculate` | MFR-003 | ✅ 완료 | 프론트엔드 연동 완료 |
| 2 | POST | `/removal/calculate` | MFR-003 | ✅ 완료 | 프론트엔드 연동 완료 |
| 3 | POST | `/analytics/parameters` | MFR-004 | ✅ 완료 | F/M비, SRT, HRT 등 |
| 4 | POST | `/analytics/reactor-distribution` | MFR-004 | ✅ 완료 | 반응조별 부하 분배 |
| 5 | POST | `/predict/{stage}` | MFR-005 | ✅ 모델 연동 완료 | Stage 02~08 검증 완료. DB 연동 필요 |
| 6 | POST | `/predict/full-pipeline` | MFR-005 | ⬜ Stub | PipelineOrchestrator 미구현 |
| 7 | GET | `/predict/history` | MFR-005 | ⬜ Stub | DB 연동 필요 |
| 8 | POST | `/effluent/predict` | MFR-005 | ⬜ Stub | 시뮬레이션 모델 연동 필요 |
| 9 | GET | `/effluent/current` | MFR-005 | ⬜ Stub | 실시간 데이터(DB) 연동 필요 |
| 10 | POST | `/anomaly/detect` | MFR-006 | 🔧 모델 연동 테스트 | Statistical + Isolation Forest |
| 11 | GET | `/anomaly/history` | MFR-006 | ⬜ Stub | DB 연동 필요 |
| 12 | POST | `/optimization/optimal-control` | MFR-007 | 🔧 모델 연동 테스트 | |
| 13 | POST | `/optimization/aeration` | MFR-008 | 🔧 모델 연동 테스트 | 송풍량 최적화 |
| 14 | POST | `/optimization/carbon` | MFR-008 | 🔧 모델 연동 테스트 | 외부탄소원 최적화 |
| 15 | POST | `/optimization/coagulant` | MFR-008 | 🔧 모델 연동 테스트 | 응집제 최적화 |

### 3.2 MFR-005 모델 연동 검증 결과

`PredictionService`를 통해 Stage별 `predict_stage()` 호출 검증 완료.

| Stage | Endpoint | 상태 | 출력 필드 |
|-------|----------|:----:|----------|
| 01 유입수 | `/predict/influent` | ✅ | BOD_부하량, TN_부하량, TP_부하량 |
| 02 일차침전지 | `/predict/primary-clarifier` | ✅ | 일차침전_BOD_eff_next, SS_eff_next |
| 03 혐기조 | `/predict/anaerobic` | ✅ | 혐기_S_PO4, MLSS, BOD |
| 04 무산소조 | `/predict/anoxic` | ✅ | 무산소_S_NH, S_NO, MLSS |
| 05 호기조 | `/predict/aerobic` | ✅ | 호기_S_NO/S_NH/MLSS/DO/BOD_next |
| 06 이차침전지 | `/predict/secondary-clarifier` | ✅ | 이차_BOD/COD/SS/TN/TP_next |
| 07 총인처리 | `/predict/tp-treatment` | ✅ | 총인처리_TP/TN_next |
| 08 방류수 | `/predict/effluent` | ✅ | 방류_TN/TP/BOD/SS/NH4/NO3/COD_next |

### 3.3 프론트엔드 연동 현황

| API Endpoint | 연동 상태 | 연동 파일 |
|-------------|----------|----------|
| `POST /load/calculate` | ✅ 완료 | dashboard, influent-integrated |
| `POST /removal/calculate` | ✅ 완료 | analysis/influent-integrated |
| 기타 22개 | ⬜ 미연동 | 순차 진행 예정 |

> 연동은 `assets/js/api-service.js`를 통해 수행. fetch 래퍼, 재시도, 타임아웃, 로컬 폴백 기능 구현.

---

## 4. DB 분석 결과

> 상세: [access-patterns.md](./access-patterns.md)

### 4.1 테이블 구조

EPSEnE DB는 MariaDB 기반이며, AI API에서 사용하는 핵심 테이블은 5개.

| 테이블 | 용도 | 현재 규모 | 일일 증가량 | 1년 후 전망 |
|--------|------|----------|-----------|-----------|
| `A2O_MEASR` | 시계열 이력 (센서 누적) | 236만건 | ~24만건/일 | **~1.1억건** |
| `A2O_MEASR_NOW` | 실시간 현재값 (UPSERT) | ~240건 | 0 (고정) | ~240건 |
| `A2O_TAG` | 태그 마스터 (임계값 등) | ~240건 | ~0 | ~240건 |
| `A2O_ALRM` | 이상탐지 이력 | ~수천건 | 변동적 | ~수만건 |
| `A2O_FCLTY_CTL` | 제어이력 | ~수백건 | ~수건 | ~수천건 |

### 4.2 API 모듈별 DB 의존성

| 모듈 | DB 의존 | 주요 테이블 | 비고 |
|------|:------:|------------|------|
| MFR-003 부하/제거율 | 없음 | - | 순수 계산 (입력값 기반) |
| MFR-004 공정분석 | 없음 | - | 순수 계산 (입력값 기반) |
| MFR-005 예측모델 | 부분 | `A2O_MEASR`, `A2O_MEASR_NOW` | 시계열 조회 + 실시간 현황 |
| MFR-006 이상탐지 | **필수** | `A2O_TAG`, `A2O_ALRM` | 임계값 조회 + 이력 INSERT |
| MFR-007 의사결정 | 예정 | `A2O_FCLTY_CTL`, `A2O_MEASR` | 제어이력 분석 |
| MFR-008 최적제어 | 예정 | `A2O_MEASR_NOW`, `A2O_FCLTY_CTL` | 실시간 조회 + 제어 INSERT |

### 4.3 데이터 흐름

```
SCADA 센서 (240+ 태그, 1~5분 주기)
    │
    ├─ 실시간 ──→ A2O_MEASR_NOW (1건/태그, UPSERT)
    │              └→ 대시보드, 이상탐지, 최적제어
    │
    ├─ 이력 ────→ A2O_MEASR (시계열 누적, INSERT)
    │              └→ 예측모델 학습/검증, 부하분석, 트렌드
    │
    └─ 이벤트 ──→ A2O_ALRM (조건 충족 시 INSERT)
                   └→ 이상탐지 이력
```

### 4.4 최적화 권고사항

| 우선순위 | 항목 | 내용 |
|:--------:|------|------|
| 필수 | A2O_MEASR 파티셔닝 | 1년 후 1.1억건. 월별 RANGE 파티셔닝 권장 |
| 강력 권장 | A2O_TAG 캐싱 | ~240건 마스터 데이터 애플리케이션 레벨 인메모리 캐싱 (TTL 5분) |
| 권장 | A2O_ALRM 인덱스 | `MEASR_DT` 단독 인덱스 + `(TAG_ID, MEASR_DT)` 복합 인덱스 추가 |

---

## 5. 프론트엔드 분석 결과

> 상세: [menu_mapping.md](./menu_mapping.md)

### 5.1 시스템 구조

```
frontend/a2o-system/
├── index.html                     # 메인 프레임 (사이드바 + iframe)
├── pages/                         # 각 화면별 HTML (22개)
│   ├── dashboard.html             # 통합 상황판
│   ├── monitoring/                # 공정 감시 (4개)
│   ├── analysis/                  # 분석·진단 (2개)
│   ├── optimization/              # 최적화 (3개)
│   ├── simulation/                # AI 시뮬레이션 (2개)
│   ├── data/                      # 데이터 관리 (5개)
│   └── system/                    # 시스템 관리 (3개)
└── assets/
    └── js/api-service.js          # API 연동 서비스 레이어
```

### 5.2 메뉴 구성 (19개 메뉴)

| 대메뉴 | 서브 메뉴 | 사용 API |
|--------|----------|---------|
| 대시보드 | 통합 상황판 | MFR-003, MFR-005, MFR-006 |
| 공정 감시 | 유입수, 생물반응조, 방류수 | MFR-003, MFR-005 |
| 분석·진단 | 유입수 분석, 생물반응조 분석 | MFR-003, MFR-004, MFR-006 |
| 최적화 | 송풍량, 외부탄소원, 응집제 | MFR-008 |
| AI 시뮬레이션 | 최적 운전제어값, 방류수 예측 | MFR-005, MFR-007 |
| 데이터 관리 | 원천데이터, 기능별, 통계, 공정분석, 다중추이 | 부록 API |
| 시스템 | 운전 목표값, 권한 관리, 시스템 로그 | AI API 범위 외 |

### 5.3 API 연동 현황

- **연동 완료**: 2개 (`/load/calculate`, `/removal/calculate`)
- **미연동**: 22개 (순차 진행 예정)
- **API 매핑 대상 외**: 3개 (시스템 관리 메뉴)

---

## 6. 이슈사항

### 6.1 DB 연동 관련

| # | 요청 사항 | 우선순위 | 관련 모듈 | 상세 |
|---|----------|:--------:|----------|------|
| 1 | **MariaDB 접속 정보 제공** | 높음 | 전체 | Host, Port, DB명, 계정 정보. 개발/스테이징 환경 분리 필요 |
| 2 | **A2O_MEASR 테이블 데이터 확인** | 높음 | MFR-005 | 실제 태그 데이터가 적재되고 있는지, 태그 ID 매핑 확인 |
| 3 | **A2O_MEASR_NOW 테이블 현행화 확인** | 높음 | MFR-005, 006 | SCADA 연동으로 실시간 UPSERT가 정상 동작하는지 확인 |
| 4 | **A2O_TAG 임계값 설정 현황** | 중간 | MFR-006 | 각 태그별 상한/하한/경고/위험 임계값이 설정되어 있는지 확인 |
| 5 | **A2O_MEASR 파티셔닝 적용 검토** | 중간 | MFR-005 | 1년 후 1.1억건 전망. 월별 RANGE 파티셔닝 필요 |

### 6.2 명세서 검토 관련

| # | 요청 사항 | 우선순위 | 관련 모듈 |
|---|----------|:--------:|----------|
| 6 | **MFR-005 Response 값 검증** | 높음 | MFR-005 |
| 7 | **MFR-006 Response 값 검증** | 높음 | MFR-006 |
| 8 | **MFR-007 명세서 검토** | 중간 | MFR-007 |
| 9 | **MFR-008 명세서 검토** | 중간 | MFR-008 |

### 6.3 프론트엔드 관련

| # | 요청 사항 | 우선순위 | 상세 |
|---|----------|:--------:|------|
| 10 | **API 연동 우선순위 합의** | 높음 | 24개 엔드포인트 중 어떤 메뉴부터 연동할지 우선순위 결정 필요 |
| 11 | **시스템 관리 메뉴 API 범위 확인** | 낮음 | 운전 목표값, 권한 관리, 시스템 로그는 AI API 범위 외. 별도 담당 확인 |

### 6.4 프론트엔드-명세서 불일치 사항

EPSEnE 프론트엔드(`frontend/a2o-system/`) 프로토타입과 기능명세서/과업지시서/databook 간의 구조적 불일치가 확인됨. 상세: [menu_mapping.md](./menu_mapping.md)

#### 6.4.1 과업지시서에 없는 API를 요청하는 페이지

프론트엔드 22개 페이지 중 아래 8개 페이지는 **과업지시서(MFR-003~008)에 포함되지 않는** 부록(FUN) 전용 API를 요청함. 해당 API 9건은 기능명세서 부록에서만 정의되어 있으며, 과업지시서의 MFR 요구사항에는 존재하지 않음.

| # | 페이지 | 요청 API | FUN 코드 | 기능 |
|---|--------|---------|----------|------|
| 1 | `dashboard.html` | `GET /dashboard/optimization-summary` | FUN-000-0100 (부록 9.1) | 최적화 권고 요약 |
| 2 | `monitoring/bioreactor.html` | `GET /bioreactor/ai-auto-status` | FUN-001-0400 (부록 9.2) | AI 자동 제어 상태 |
| 3 | `analysis/influent-integrated.html` | `POST /analytics/influent-analysis` | FUN-002-0100 (부록 9.3) | 부하 상태, 군집 판정, AI 진단 |
| 4 | `analysis/bioreactor-analysis.html` | `POST /analytics/bioreactor-analysis` | FUN-002-0200 (부록 9.4) | SVI 등급, 안정성 진단, 벌킹 경고 |
| 5 | `data/raw-data.html` | `GET/POST /data/raw` | FUN-005-0100 (부록) | 원천데이터 조회 (102개 항목) |
| 6 | `data/function-data.html` | `GET /data/function/{function_name}` | FUN-005-0200 (부록) | 기능별 데이터 조회 |
| 7 | `data/multi-trend.html` | `POST /data/multi-trend` | FUN-005-0300 (부록) | 다중 항목 추이 + 상관계수 |
| 8 | `data/process-analysis.html` | `POST /analytics/correlation` | FUN-005-0400 (부록) | 히트맵, 산점도, 군집 레이더 |
| 9 | `data/statistics.html` | `POST /analytics/statistics` | FUN-005-0500 (부록) | 기술통계, Q-Q플롯 |

> **참고**: 위 9건의 API는 **과업지시서에 없으며**, 기능명세서 부록(FUN-000~005)에서만 정의되어 있음. 프론트엔드가 해당 기능을 탑재하고 있으므로, 상호 협의하여 추가 개발 범위에 포함할지 결정이 필요함.

#### 6.4.2 과업지시서 vs 기능명세서 API 범위 비교

| 구분 | 엔드포인트 수 | 출처 | 과업지시서 포함 |
|------|:----------:|------|:-------------:|
| MFR API (MFR-003~008) | **15개** | 과업지시서 + 기능명세서 | O |
| 부록 API (FUN-000~005) | **9개** | 기능명세서 부록에만 존재 | **X** |
| 합계 | **24개** | | |

> **핵심 요청**: 프론트엔드가 24개 API를 전제로 설계되어 있으나, 과업지시서 기반 개발 범위는 15개(MFR)임. 나머지 9개(부록 FUN) API는 과업지시서에 포함되지 않으므로, 상호 협의하여 추가 개발 여부 및 우선순위를 결정해야 할 것으로 판단됨.
