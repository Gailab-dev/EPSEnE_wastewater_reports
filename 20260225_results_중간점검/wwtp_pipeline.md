# WWTP Pipeline

하수처리장 8단계 순차 예측 AI 파이프라인

## 개요

이 패키지는 하수처리장의 유입수(Influent)부터 방류수(Effluent)까지 8개 공정 단계를 순차적으로 예측하는 AI 파이프라인을 제공합니다.

### 8단계 공정

| Stage | 공정 | 주요 예측 | 상태 |
|-------|------|----------|------|
| 1 | 유입수 (Influent) | BOD/TN/TP 부하량 예측 | ✅ 구현 |
| 2 | 일차침전지 (Primary Clarifier) | 침전 후 수질 (BOD, TOC, SS, TN, TP) | ✅ 구현 |
| 3 | 혐기조 (Anaerobic) | 인 방출 (PO4-P, MLSS) | ✅ 구현 |
| 4 | 무산소조 (Anoxic) | 탈질 (NH4, NO3, MLSS) | ✅ 구현 |
| 5 | 호기조 (Aerobic) | 질산화 및 인 흡수 (NH4, NO3, PO4-P, MLSS) | ✅ 구현 |
| 6 | 이차침전지 (Secondary Clarifier) | 슬러지 반송/인발 (Q_RAS, Q_WAS, HRT) | ✅ 구현 |
| 7 | 총인처리시설 (P Treatment) | 화학적 인 제거 (방류_TP, 약품주입률) | ✅ 구현 |
| 8 | 방류수 (Effluent) | 최종 방류수 수질 (BOD, TOC, SS, TN, TP) | ✅ 구현 |

---

## 설치

```bash
# 개발 모드 설치
pip install -e .

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

---

## 사용법 [미구현]

> **주의**: 아래 사용법은 `PipelineOrchestrator`가 구현된 후 사용 가능합니다.

### 기본 사용

```python
from wwtp_pipeline import PipelineOrchestrator
from wwtp_pipeline.schemas import PipelineInput
from datetime import datetime

# 파이프라인 초기화
orchestrator = PipelineOrchestrator()

# 입력 데이터 생성
input_data = PipelineInput(
    timestamp=datetime.now(),
    유입유량=128000,
    유입BOD=156.8,
    유입TN=24.9,
    유입TP=4.85,
    유입TOC=49.31,
    유입SS=142.0,
    수온=15.5,
    pH=7.2
)

# 예측 실행
result = orchestrator.predict(input_data)

# 결과 확인
print(f"방류_BOD: {result.방류_BOD:.2f} mg/L")
print(f"방류_TN: {result.방류_TN:.2f} mg/L")
print(f"방류_TP: {result.방류_TP:.4f} mg/L")
print(f"법적 기준 준수: {result.compliance_check}")
```

### 배치 예측

```python
# 여러 샘플 동시 예측
input_batch = [PipelineInput(...) for _ in range(100)]
results = orchestrator.predict_batch(input_batch)
```

---

## 프로젝트 구조

```
wwtp_pipeline/
├── __init__.py
├── pyproject.toml
├── requirements.txt
│
├── core/
│   ├── orchestrator.py          # 8단계 순차 실행 [미구현]
│   ├── exceptions.py
│   ├── feature_engineer.py
│   └── multi_output_regressor.py
│
├── stages/
│   ├── base.py                  # BaseStage 추상 클래스
│   ├── stage_01_influent.py
│   ├── stage_02_primary_clarifier.py
│   ├── stage_03_anaerobic.py
│   ├── stage_04_anoxic.py       # [미구현]
│   ├── stage_05_aerobic.py      # [미구현]
│   ├── stage_06_secondary_clarifier.py  # [미구현]
│   ├── stage_07_phosphorus_removal.py   # [미구현]
│   └── stage_08_effluent.py     # [미구현]
│
├── schemas/
│   ├── input.py                 # PipelineInput
│   ├── output.py                # Stage outputs, PipelineOutput
│   └── intermediate.py          # IntermediateFeatures
│
├── loaders/                     # 모델 로더
├── config/                      # 설정
├── utils/
│   └── preprocessing.py
│
├── models/                      # 학습된 모델 파일
│   ├── 01_influent/
│   └── 02_primary_clarifier/
│
└── tests/
    ├── unit/
    └── integration/
```

---

## 성능 목표

| 항목 | 목표값 |
|------|--------|
| 단일 예측 | < 1초 |
| 배치 예측 (100개) | < 10초 |
| 모델 정확도 | R² ≥ 0.85, MAPE ≤ 15% |
| End-to-End 정확도 | R² ≥ 0.80 |

---

## 방류수 법적 기준 (I지역)

| 항목 | 기준 (mg/L) |
|------|-------------|
| BOD | ≤ 5 |
| COD | ≤ 40 |
| SS | ≤ 10 |
| TN | ≤ 20 |
| TP | ≤ 0.2 |

> **참고**: databook (design_capacity.md, databook_measurable.md) 기준

---

## 라이선스

MIT License
