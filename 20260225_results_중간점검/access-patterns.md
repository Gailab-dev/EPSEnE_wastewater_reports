# DB 접근 패턴 분석

**문서번호**: EPSEnE-DB-ACCESS-v1.0
**작성일자**: 2026-02-26
**관련 문서**: [DB 스키마 명세서](./schemas.md), [API 명세서](../api.md)

---

## 1. 데이터 흐름

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

---

## 2. API 모듈별 DB 의존성

| 모듈 | DB 의존 | 주요 테이블 | 접근 유형 | 빈도 |
|------|---------|------------|----------|------|
| **MFR-003** 부하/제거율 | 없음 | - | 순수 계산 (입력값 기반) | - |
| **MFR-004** 공정분석 | 없음 | - | 순수 계산 (입력값 기반) | - |
| **MFR-005** 예측모델 | 부분 | `A2O_MEASR`, `A2O_MEASR_NOW`, `A2O_TAG` | 시계열 범위조회 + PK 조회 | 높음 |
| **MFR-006** 이상탐지 | 필수 | `A2O_TAG`, `A2O_ALRM` | PK 조회 + 범위조회 + INSERT | 매우 높음 |
| **MFR-007** 의사결정 | 예정 | `A2O_FCLTY_CTL`, `A2O_MEASR` | 이력 분석 | 중간 |
| **MFR-008** 최적제어 | 예정 | `A2O_MEASR_NOW`, `A2O_FCLTY_CTL` | PK 조회 + INSERT | 중간 |

---

## 3. 핵심 접근 패턴

### 패턴 A: 실시간 현황 조회 (Very High Frequency)

**사용처**: `/effluent/current`, `/anomaly/detect`, 대시보드 갱신 (30초~1분 주기)

```sql
SELECT TAG_ID, MEASR_CVL, MEASR_DT, CTL_GOAL_CVL
FROM A2O_MEASR_NOW
WHERE TAG_ID IN ('방류_BOD', '방류_COD', '방류_TN', '방류_TP', '방류_SS');
```

- **테이블**: `A2O_MEASR_NOW` — TAG_ID당 1건, PK 조회
- **특성**: 초경량, 태그 수만큼 행 존재 (~240건)
- **추가 조치 필요 없음** (PK 조회라 이미 최적)

### 패턴 B: 시계열 범위 조회 (High Frequency)

**사용처**: `/predict/history`, 30일 트렌드 분석

```sql
SELECT MEASR_CVL, MEASR_DT
FROM A2O_MEASR
WHERE TAG_ID = ? AND MEASR_DT BETWEEN ? AND ?
ORDER BY MEASR_DT DESC;
```

- **테이블**: `A2O_MEASR` — 236만건+, 일 ~24만건 증가
- **인덱스**: `idx_A2O_MEASR_TAG_ID_MEASR_DT` (복합) — 이미 존재
- **추가 조치**: 파티셔닝 검토 필요 (섹션 5 참조)

### 패턴 C: 임계값 기반 이상탐지 (Very High Frequency)

**사용처**: `/anomaly/detect` 호출 시

```sql
SELECT TAG_ID, UPLMT_CVL, LLMT_CVL,
       WRNNG_UPLMT_CVL, WRNNG_LLMT_CVL,
       RISK_UPLMT_CVL, RISK_LLMT_CVL
FROM A2O_TAG
WHERE PROCS_CD = ? AND DEL_YN = 'N';
```

- **테이블**: `A2O_TAG` — ~240건, 거의 변하지 않는 마스터
- **추가 조치**: 애플리케이션 캐싱 강력 권장 (섹션 5 참조)

### 패턴 D: 이상탐지 이력 조회 (Medium Frequency)

**사용처**: `/anomaly/history`

```sql
SELECT a.*, t.TAG_NM, t.PROCS_CD
FROM A2O_ALRM a
JOIN A2O_TAG t ON a.TAG_ID = t.TAG_ID
WHERE a.MEASR_DT BETWEEN ? AND ?
  AND (? IS NULL OR t.PROCS_CD = ?)
ORDER BY a.MEASR_DT DESC
LIMIT ? OFFSET ?;
```

- **테이블**: `A2O_ALRM` JOIN `A2O_TAG`
- **인덱스**: 현재 TAG_ID FK만 존재 — MEASR_DT 인덱스 추가 권장

### 패턴 E: 제어이력 분석 (Low~Medium Frequency)

**사용처**: MFR-007 의사결정, MFR-008 최적제어

```sql
SELECT FCLTY_CD, BFR_CVL, AFTR_CVL, CTL_DT, CTL_CN
FROM A2O_FCLTY_CTL
WHERE FCLTY_CD = ? AND CTL_DT BETWEEN ? AND ?
ORDER BY CTL_DT DESC;
```

- **테이블**: `A2O_FCLTY_CTL`
- **특성**: 제어 변경 시에만 INSERT, 저빈도

---

## 4. 데이터 볼륨 전망

| 테이블 | 현재 | 일일 증가량 | 1년 후 | 비고 |
|--------|------|-----------|--------|------|
| `A2O_MEASR` | 236만 | **~24만건/일** | **~1.1억건** | 240태그 x 1000회/일 |
| `A2O_MEASR_NOW` | ~240 | 0 (UPSERT) | ~240 | 고정 크기 |
| `A2O_TAG` | ~240 | ~0 | ~240 | 마스터 |
| `A2O_ALRM` | ~수천 | 변동적 | ~수만 | 이상 발생 시만 |
| `A2O_FCLTY_CTL` | ~수백 | ~수건 | ~수천 | 제어 변경 시만 |

> **A2O_MEASR가 압도적 대용량**. 1년 후 1억건 이상이므로 이 테이블의 성능이 전체 시스템 성능을 결정.

---

## 5. 최적화 전략

### 5.1 A2O_MEASR 파티셔닝 (필수 검토)

1억건 이상 시 범위 조회 성능이 급격히 저하됨. 월별 RANGE 파티셔닝을 권장.

```sql
ALTER TABLE A2O_MEASR
PARTITION BY RANGE (UNIX_TIMESTAMP(MEASR_DT)) (
    PARTITION p202601 VALUES LESS THAN (UNIX_TIMESTAMP('2026-02-01')),
    PARTITION p202602 VALUES LESS THAN (UNIX_TIMESTAMP('2026-03-01')),
    ...
);
```

**효과**:
- 날짜 범위 쿼리 시 해당 파티션만 스캔 (partition pruning)
- 오래된 데이터 아카이빙/삭제를 파티션 단위로 처리 가능
- 인덱스 크기 감소로 전체적인 쿼리 성능 향상

### 5.2 A2O_TAG 캐싱 (강력 권장)

~240건의 마스터 데이터를 이상탐지마다 반복 조회하는 것은 비효율적.

```python
# 애플리케이션 레벨 인메모리 캐싱
tag_cache: dict[str, list[TagThresholds]] = {}  # TTL: 5분

async def get_tag_thresholds(procs_cd: str) -> list[TagThresholds]:
    if procs_cd in tag_cache and not expired:
        return tag_cache[procs_cd]
    result = await db.execute(...)
    tag_cache[procs_cd] = result
    return result
```

**효과**: 이상탐지 호출 시 DB 조회 제거 (30초마다 → 5분마다로 감소)

### 5.3 A2O_ALRM 인덱스 보강 (권장)

현재 TAG_ID FK 인덱스(`R_6`)만 존재. `/anomaly/history`의 날짜 범위 + 필터 조회에 대응 필요.

```sql
CREATE INDEX idx_A2O_ALRM_MEASR_DT ON A2O_ALRM (MEASR_DT);
CREATE INDEX idx_A2O_ALRM_TAG_MEASR_DT ON A2O_ALRM (TAG_ID, MEASR_DT);
```

### 5.4 SCADA 수집 UPSERT 전략

```
SCADA 수집 (1분~5분 간격)
    │
    ├─ A2O_MEASR_NOW: REPLACE INTO 또는 ON DUPLICATE KEY UPDATE
    │   → 항상 최신 1건만 유지
    │
    └─ A2O_MEASR: INSERT (append-only)
        → 이력 무한 축적, FK 없음 (대량 INSERT 성능 확보)
```

> `A2O_MEASR`에 FK 제약이 없는 이유: 대량 INSERT 성능 확보 목적. TAG_ID 유효성은 SCADA 수집기(애플리케이션)에서 검증.

---

## 6. 읽기/쓰기 분류

| 접근 모드 | 테이블 | 주체 |
|----------|--------|------|
| **Read-Only** (API) | `A2O_MEASR`, `A2O_TAG`, `A2O_PROCS_CD`, `A2O_PROCS_SRS`, `A2O_FCLTY_CD` | API 서버 |
| **Read (API) + Write (SCADA)** | `A2O_MEASR_NOW` | SCADA→UPSERT, API→SELECT |
| **Write-Heavy** | `A2O_MEASR` | SCADA→INSERT, API→SELECT |
| **Append (API)** | `A2O_ALRM` | 이상탐지 시 INSERT |
| **Append (API)** | `A2O_FCLTY_CTL` | 제어 변경 시 INSERT |

> API 서버는 대부분 **읽기 전용**. 쓰기는 SCADA 수집기와 이상탐지/제어 모듈에서만 발생.

---

## 7. API 개발 시 DB 접근 우선순위

```
┌──────────────────────────────────────────────────────┐
│  우선순위 HIGH — API 구현에 즉시 필요                    │
│                                                       │
│  1. A2O_MEASR_NOW  → PK SELECT (실시간 대시보드)        │
│  2. A2O_MEASR      → 범위 SELECT (예측 이력/트렌드)     │
│  3. A2O_TAG        → SELECT + 캐싱 (이상탐지 임계값)    │
│  4. A2O_ALRM       → SELECT + INSERT (이상탐지 이력)    │
├──────────────────────────────────────────────────────┤
│  우선순위 MEDIUM — 추후 구현 시 필요                     │
│                                                       │
│  5. A2O_FCLTY_CTL  → SELECT + INSERT (제어이력)         │
│  6. A2O_PROCS_CD   → SELECT (공정 매핑)                 │
├──────────────────────────────────────────────────────┤
│  우선순위 LOW — AI API에서 직접 사용 안 함               │
│                                                       │
│  7. CM_USER / CM_AUTH_TOKEN (인증은 별도 시스템 담당)     │
│  8. 나머지 CM_* 테이블 (프론트엔드 플랫폼 기능)          │
└──────────────────────────────────────────────────────┘
```
