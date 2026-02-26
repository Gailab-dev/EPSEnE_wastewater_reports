# 📋 메뉴-파일-API 매핑

> **기준일**: 2026-02-24
> **API Base URL**: `/api/v1`
> **API 명세서**: `docs/api.md`

---

## 📊 대시보드

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 통합 상황판 | `pages/dashboard.html` | `/dashboard/optimization-summary` | GET | FUN-000-0100 (부록 9.1) | 최적화 권고 요약 | - |
| | | `/effluent/current` | GET | MFR-005 / FUN-001-0500 | 방류수 실시간 현황 | - |
| | | `/anomaly/detect` | POST | MFR-006 | 이상탐지 상태 | - |
| | | `/load/calculate` | POST | MFR-003 / FUN-001-0300 | 부하율 현황 | 완료 |

---

## 💧 공정 감시 (4개 메뉴)

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 유입수 감시 | `pages/monitoring/influent-integrated.html` | `/predict/influent` | POST | MFR-005 / FUN-001-0100 | 유입수 예측 (Stage 01) | - |
| | *(탭: 유입유량)* | `/load/calculate` | POST | MFR-003 / FUN-001-0300 | 부하량 산정 | 완료 |
| | *(탭: 유입수질)* | `/predict/influent` | POST | MFR-005 / FUN-001-0200 | 유입수질 추이 | - |
| | *(탭: 유입부하)* | `/load/calculate` | POST | MFR-003 / FUN-001-0300 | 부하량·부하율 | 완료 |
| 생물반응조 | `pages/monitoring/bioreactor.html` | `/predict/{stage}` | POST | MFR-005 / FUN-001-0400 | stage: anaerobic, anoxic, aerobic | - |
| | | `/bioreactor/ai-auto-status` | GET | FUN-001-0400 (부록 9.2) | AI 자동 제어 상태 | - |
| | | `/analytics/parameters` | POST | MFR-004 | F/M비, SRT, SVI 등 | - |
| 방류수 | `pages/monitoring/effluent.html` | `/effluent/current` | GET | MFR-005 / FUN-001-0500 | 실시간 방류수질 + 기준 달성률 | - |
| | | `/predict/effluent` | POST | MFR-005 | 방류수질 단기 예측 (1h/24h) | - |

> **참고**: MENU_MAPPING.md(v1)에 있던 `influent-flow.html`, `influent-quality.html`, `influent-load.html`은 실제로 존재하지만, 사이드바 메뉴에서는 `influent-integrated.html`이 유입수 감시 단일 진입점으로 사용됩니다. 개별 파일은 탭 또는 서브페이지 역할입니다.

---

## 📈 분석·진단 (2개 메뉴)

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 유입수 분석 | `pages/analysis/influent-integrated.html` | `/analytics/influent-analysis` | POST | FUN-002-0100 (부록 9.3) | 부하 상태, 군집 판정, AI 진단 | - |
| | | `/load/calculate` | POST | MFR-003 | 부하율 산정 | 완료 |
| | | `/removal/calculate` | POST | MFR-003 | 제거율 산정 | 완료 |
| 생물반응조 분석 | `pages/analysis/bioreactor-analysis.html` | `/analytics/bioreactor-analysis` | POST | FUN-002-0200 (부록 9.4) | SVI 등급, 안정성 진단, 벌킹 경고 | - |
| | | `/analytics/parameters` | POST | MFR-004 | 운영인자 계산 (F/M, SRT, HRT) | 완료 |
| | | `/analytics/reactor-distribution` | POST | MFR-004 | 반응조별 부하 분배 | 완료 |
| | | `/anomaly/detect` | POST | MFR-006 | 공정별 이상탐지 | - |

---

## 💨 최적화 (3개 메뉴)

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 송풍량 최적화 | `pages/optimization/aeration.html` | `/optimization/aeration` | POST | MFR-008 / FUN-003-0100 | AI 추천 송풍량, DO 예측 36h, 비용 | - |
| 외부탄소원 | `pages/optimization/chemical-carbon.html` | `/optimization/carbon` | POST | MFR-008 / FUN-003-0200 | AI 추천 탄소원량, TN 예측 36h, 비용 | - |
| 응집제 | `pages/optimization/chemical-polymer.html` | `/optimization/coagulant` | POST | MFR-008 / FUN-003-0300 | AI 추천 응집제량, TP 예측 36h, 비용 | - |

---

## 🤖 AI 시뮬레이션 (2개 메뉴)

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 최적 운전제어값 | `pages/simulation/optimal-control.html` | `/optimization/optimal-control` | POST | MFR-007 / FUN-004-0100 | 통합 최적제어값 6개 항목 도출 | - |
| 방류수 예측 | `pages/simulation/effluent-prediction.html` | `/effluent/predict` | POST | MFR-005 / FUN-004-0200 | 방류수질 시뮬레이션 (72h, 6h 간격) | - |
| | | `/predict/full-pipeline` | POST | MFR-005 | 전체 파이프라인 예측 (선택) | - |

---

## 💾 데이터 관리 (5개 메뉴)

| 메뉴 | 파일 | API Endpoint | Method | MFR/FUN | 비고 | 연동여부 |
|------|------|-------------|--------|---------|------|----------|
| 원천데이터 조회 | `pages/data/raw-data.html` | `/data/raw` | GET/POST | FUN-005-0100 (부록) | 102개 항목, 최대 1년 | - |
| 기능별 데이터 | `pages/data/function-data.html` | `/data/function/{function_name}` | GET | FUN-005-0200 (부록) | 기능별 데이터 조회 | - |
| 계측요소별 통계 | `pages/data/statistics.html` | `/analytics/statistics` | POST | FUN-005-0500 (부록) | 기술통계, Q-Q플롯 | - |
| 공정별 분석 | `pages/data/process-analysis.html` | `/analytics/correlation` | POST | FUN-005-0400 (부록) | 히트맵, 산점도, 군집 레이더 | - |
| 다중 항목 추이 | `pages/data/multi-trend.html` | `/data/multi-trend` | POST | FUN-005-0300 (부록) | 다중 추이 + 상관계수 | - |

---

## ⚙️ 시스템 (3개 메뉴) — API 매핑 대상 외

| 메뉴 | 파일 | 비고 |
|------|------|------|
| 운전 목표값 | `pages/system/target-settings.html` | AI API 범위 외 — 별도 시스템 API 필요 |
| 권한 관리 | `pages/system/roles.html` | AI API 범위 외 — 인증/인가 시스템 |
| 시스템 로그 | `pages/system/logs.html` | AI API 범위 외 — 로그 조회 시스템 |

> MENU_MAPPING.md(v1)에 있던 `pages/system/users.html`(사용자 관리)은 실제 파일이 존재하지 않습니다.

---

## 🎯 총계

| 구분 | 수량 |
|------|------|
| 사이드바 메뉴 항목 | 19개 |
| 실제 HTML 파일 | 22개 (메뉴 19 + 서브페이지 3) |
| 연동 API 엔드포인트 (MFR) | 15개 |
| 연동 API 엔드포인트 (부록) | 9개 |
| API 매핑 대상 외 (시스템) | 3개 |

---

## 📎 파일 불일치 메모

### MENU_MAPPING.md(v1) 대비 변경 사항

**삭제된 메뉴 (실제 파일 없음 또는 통합됨)**:
- `pages/analysis/influent-variability.html` — 실제 파일 없음 → `influent-integrated.html`로 통합
- `pages/analysis/influent-load-analysis.html` — 실제 파일 없음 → `influent-integrated.html`로 통합
- `pages/analysis/influent-pattern.html` — 실제 파일 없음 → `influent-integrated.html`로 통합
- `pages/simulation/asm2d.html` — 실제 파일 없음 → `effluent-prediction.html` + `optimal-control.html`로 대체
- `pages/system/users.html` — 실제 파일 없음

**추가된 메뉴 (v1에 없었으나 실제 존재)**:
- `pages/monitoring/influent-integrated.html` — 유입수 감시 통합 진입점 (사이드바 메뉴)
- `pages/simulation/effluent-prediction.html` — 방류수 예측 시뮬레이션
- `pages/simulation/optimal-control.html` — 최적 운전제어값

**서브페이지 (사이드바 메뉴에는 없으나 존재하는 파일)**:
- `pages/monitoring/influent-flow.html` — influent-integrated 내 탭/서브
- `pages/monitoring/influent-quality.html` — influent-integrated 내 탭/서브
- `pages/monitoring/influent-load.html` — influent-integrated 내 탭/서브

---

## 📐 API 엔드포인트 ↔ 페이지 역매핑

| # | API Endpoint | Method | 사용 페이지 | 연동여부 |
|---|-------------|--------|-----------|----------|
| 1 | `/load/calculate` | POST | dashboard, influent-integrated, analysis/influent-integrated | 완료 |
| 2 | `/removal/calculate` | POST | analysis/influent-integrated | 완료 |
| 3 | `/analytics/parameters` | POST | analysis/bioreactor-analysis | 완료 |
| 4 | `/analytics/reactor-distribution` | POST | analysis/bioreactor-analysis | 완료 |
| 5 | `/predict/full-pipeline` | POST | simulation/effluent-prediction | - |
| 6 | `/predict/{stage}` | POST | monitoring/bioreactor | - |
| 7 | `/predict/history` | GET | *(사용 페이지 미확정)* | - |
| 8 | `/effluent/predict` | POST | simulation/effluent-prediction | - |
| 9 | `/effluent/current` | GET | dashboard, monitoring/effluent | - |
| 10 | `/anomaly/detect` | POST | dashboard, analysis/bioreactor-analysis | - |
| 11 | `/anomaly/history` | GET | *(사용 페이지 미확정 — 시스템 로그 또는 대시보드 이력)* | - |
| 12 | `/optimization/optimal-control` | POST | simulation/optimal-control | - |
| 13 | `/optimization/aeration` | POST | optimization/aeration | - |
| 14 | `/optimization/carbon` | POST | optimization/chemical-carbon | - |
| 15 | `/optimization/coagulant` | POST | optimization/chemical-polymer | - |
| 16 | `/analytics/statistics` | POST | data/statistics | - |
| 17 | `/analytics/correlation` | POST | data/process-analysis | - |
| 18 | `/dashboard/optimization-summary` | GET | dashboard | - |
| 19 | `/bioreactor/ai-auto-status` | GET | monitoring/bioreactor | - |
| 20 | `/analytics/influent-analysis` | POST | analysis/influent-integrated | - |
| 21 | `/analytics/bioreactor-analysis` | POST | analysis/bioreactor-analysis | - |
| 22 | `/data/raw` | GET/POST | data/raw-data | - |
| 23 | `/data/function/{function_name}` | GET | data/function-data | - |
| 24 | `/data/multi-trend` | POST | data/multi-trend | - |

> **미배정 API**: `#7 /predict/history`, `#11 /anomaly/history` — 현재 프론트엔드에 전용 페이지 없음. 대시보드 이력 탭 또는 시스템 로그에서 활용 가능.

---

## 🔗 API 연동 현황

> **기준일**: 2026-02-24

| # | API Endpoint | MFR | 연동 상태 | 연동 파일 | 비고 |
|---|-------------|-----|----------|----------|------|
| 1 | `POST /load/calculate` | MFR-003 | ✅ 완료 | `dashboard.html`, `monitoring/influent-load.html`, `analysis/influent-integrated.html` | `api-service.js` 사용, 폴백 지원 |
| 2 | `POST /removal/calculate` | MFR-003 | ✅ 완료 | `analysis/influent-integrated.html` | `api-service.js` 사용, 폴백 지원 |
| 3~24 | 기타 | - | ⬜ 미연동 | - | 순차 연동 예정 |

### API 서비스 레이어

- **파일**: `assets/js/api-service.js`
- **Base URL**: `/api/v1` (설정 변경 가능: `ApiService.configure({ BASE_URL: '...' })`)
- **기능**: fetch 래퍼, 재시도, 타임아웃, 로컬 폴백
- **MFR-003 메서드**: `ApiService.calculateLoad()`, `ApiService.calculateRemoval()`
