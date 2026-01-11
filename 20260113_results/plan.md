# 구현 계획: 하수처리장 AI API 서비스

**Branch**: `001-wwtp-ai-api` | **Date**: 2026-01-07 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-wwtp-ai-api/spec.md`

## 요약

하수처리장 운영을 위한 종합 AI API 서비스 개발. 실시간 공정 모니터링, 이상 탐지, 유입부하 분석, 공정 시뮬레이션, 의사결정 지원, 제어인자 최적화 기능을 제공하는 FastAPI 기반 마이크로서비스 아키텍처로 구현. PyTorch 기반 AI 모델을 활용하여 95% 이상의 정밀도와 재현율을 목표로 하는 고성능 이상 탐지 시스템 구축.

## Technical Context

**Language/Version**: Python 3.12.3

**Primary Dependencies**: FastAPI, PyTorch, Pydantic, SQLAlchemy, scikit-learn, XGBoost, pandas, numpy

**Storage**: MariaDB (운영 데이터, 계산 결과, 의사결정 이력), 파일 시스템 (훈련된 모델 저장)

**Testing**: pytest, pytest-asyncio, httpx (API 테스트)

**Target Platform**: Linux server (Docker 컨테이너)


**Project Type**: Web API (백엔드 전용, RESTful API)

**Performance Goals**:
- API 응답시간: 부하 계산 <10초, 시뮬레이션 <30초, 최적화 계산 <20초
- 이상 탐지: 실시간 처리 5분 이내 탐지
- 동시 처리: 최소 50개 이상의 수질 파라미터 분석
- API 가용성: 99.5% uptime

**Constraints**:
- 이상 탐지 성능: Precision ≥ 95%, Recall ≥ 95%, F1-Score ≥ 0.85
- 메모리: 모델 로딩 및 추론에 효율적인 메모리 사용
- 동시성: 비동기 처리로 다중 요청 동시 처리
- 데이터 무결성: 모든 계산 결과 및 의사결정 이력 영구 저장

**Scale/Scope**:
- 9개 처리 단계 (유입수→일차침전지→혐기조→무산소조→호기조→이차침전지/분리막→총인처리→방류수)
- 50+ 수질 파라미터 동시 모니터링
- 38개 기능 요구사항
- 6가지 주요 제어 파라미터 최적화

## Constitution Check

*GATE: Phase 0 연구 전 통과 필요. Phase 1 설계 후 재확인.*

**Note**: 프로젝트 Constitution이 아직 정의되지 않았으므로, 다음 기본 원칙을 적용:

### 기본 개발 원칙

1. **모듈화**: 각 기능 영역 (부하 분석, 시뮬레이션, 이상 탐지 등)을 독립적인 모듈로 구성
2. **테스트 가능성**: 모든 핵심 로직에 대한 단위 테스트 및 통합 테스트 작성
3. **API 우선 설계**: 명확한 계약(contract)을 통한 API 엔드포인트 정의
4. **성능**: 명시된 응답 시간 목표 달성
5. **데이터 무결성**: 모든 계산 결과 및 이력 데이터 영구 저장

**초기 평가**: ✅ PASS
- 단일 백엔드 프로젝트로 복잡도 관리 가능
- FastAPI의 자동 문서화 및 검증 활용
- PyTorch 기반 모델 추론 최적화
- docker-compose로 간단한 배포

## Project Structure

### Documentation (this feature)

```text
specs/001-wwtp-ai-api/
├── plan.md              # 이 파일 (/speckit.plan 명령 출력)
├── research.md          # Phase 0 출력 (/speckit.plan 명령)
├── data-model.md        # Phase 1 출력 (/speckit.plan 명령)
├── quickstart.md        # Phase 1 출력 (/speckit.plan 명령)
├── contracts/           # Phase 1 출력 (/speckit.plan 명령)
│   ├── load-analysis.yaml
│   ├── process-analysis.yaml
│   ├── simulation.yaml
│   ├── anomaly-detection.yaml
│   ├── decision-support.yaml
│   └── control-optimization.yaml
└── tasks.md             # Phase 2 출력 (/speckit.tasks 명령 - /speckit.plan에서 생성 안 함)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/                      # FastAPI 라우터 및 엔드포인트
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── load_analysis.py        # FR-001~005: 유입부하/제거율 API
│   │   │   ├── process_analysis.py     # FR-006~012: 공정분석 API
│   │   │   ├── simulation.py           # FR-013~018: 시뮬레이션 API
│   │   │   ├── anomaly_detection.py    # FR-019~024: 이상탐지 API
│   │   │   ├── decision_support.py     # FR-025~031: 의사결정 지원 API
│   │   │   └── control_optimization.py # FR-032~038: 제어 최적화 API
│   │   ├── dependencies.py       # 공통 의존성 (DB 세션, 인증 등)
│   │   └── middleware.py         # 로깅, CORS, 예외 처리
│   │
│   ├── models/                   # 데이터 모델 (ORM)
│   │   ├── __init__.py
│   │   ├── treatment_process.py  # 처리 공정 단계 모델
│   │   ├── water_quality.py      # 수질 측정 데이터 모델
│   │   ├── load_calculation.py   # 부하 계산 결과 모델
│   │   ├── removal_efficiency.py # 제거율 결과 모델
│   │   ├── process_indicator.py  # 공정 지표 모델
│   │   ├── anomaly_event.py      # 이상 탐지 이벤트 모델
│   │   ├── decision_recommendation.py  # 의사결정 추천 모델
│   │   ├── control_setpoint.py   # 제어 설정값 모델
│   │   └── simulation_result.py  # 시뮬레이션 결과 모델
│   │
│   ├── schemas/                  # Pydantic 스키마 (request/response)
│   │   ├── __init__.py
│   │   ├── load_analysis.py
│   │   ├── process_analysis.py
│   │   ├── simulation.py
│   │   ├── anomaly_detection.py
│   │   ├── decision_support.py
│   │   └── control_optimization.py
│   │
│   ├── services/                 # 비즈니스 로직 및 계산 엔진
│   │   ├── __init__.py
│   │   ├── load_calculator.py    # FR-001, FR-002: 부하 계산 로직
│   │   ├── removal_analyzer.py   # FR-002: 제거율 분석
│   │   ├── statistical_analyzer.py # FR-006~010: 통계 분석, 군집/상관/PCA
│   │   ├── process_simulator.py  # FR-013~018: 공정 시뮬레이션 엔진
│   │   ├── anomaly_detector.py   # FR-019~024: 이상 탐지 엔진
│   │   ├── decision_engine.py    # FR-025~031: 의사결정 엔진
│   │   └── control_optimizer.py  # FR-032~038: 제어 파라미터 최적화
│   │
│   ├── ml/                       # AI/ML 모델 관리
│   │   ├── __init__.py
│   │   ├── models/               # PyTorch 모델 정의
│   │   │   ├── __init__.py
│   │   │   ├── stage_predictor.py      # 각 공정 단계별 예측 모델
│   │   │   ├── anomaly_classifier.py   # 이상 탐지 분류기
│   │   │   └── control_predictor.py    # 제어 파라미터 예측 모델
│   │   ├── trainers/             # 모델 학습 스크립트
│   │   │   ├── __init__.py
│   │   │   ├── stage_trainer.py
│   │   │   ├── anomaly_trainer.py
│   │   │   └── control_trainer.py
│   │   ├── loaders/              # 모델 로딩 및 캐싱
│   │   │   ├── __init__.py
│   │   │   └── model_loader.py
│   │   └── preprocessors/        # 데이터 전처리
│   │       ├── __init__.py
│   │       ├── scaler.py
│   │       └── feature_engineer.py
│   │
│   ├── db/                       # 데이터베이스 관련
│   │   ├── __init__.py
│   │   ├── base.py               # SQLAlchemy Base
│   │   ├── session.py            # DB 세션 관리
│   │   └── migrations/           # Alembic 마이그레이션
│   │
│   ├── core/                     # 공통 설정 및 유틸리티
│   │   ├── __init__.py
│   │   ├── config.py             # 환경 설정 (Pydantic Settings)
│   │   ├── logging.py            # 로깅 설정
│   │   ├── exceptions.py         # 커스텀 예외
│   │   └── constants.py          # 상수 정의 (처리 단계, 파라미터 등)
│   │
│   ├── utils/                    # 유틸리티 함수
│   │   ├── __init__.py
│   │   ├── validators.py         # 데이터 유효성 검사
│   │   ├── calculators.py        # 공통 계산 함수 (F/M, C/N 등)
│   │   └── formatters.py         # 응답 포맷팅
│   │
│   └── main.py                   # FastAPI 애플리케이션 엔트리포인트
│
├── tests/
│   ├── contract/                 # API 계약 테스트
│   │   ├── test_load_analysis_api.py
│   │   ├── test_process_analysis_api.py
│   │   ├── test_simulation_api.py
│   │   ├── test_anomaly_detection_api.py
│   │   ├── test_decision_support_api.py
│   │   └── test_control_optimization_api.py
│   │
│   ├── integration/              # 통합 테스트
│   │   ├── test_end_to_end_flow.py
│   │   ├── test_db_operations.py
│   │   └── test_ml_inference.py
│   │
│   ├── unit/                     # 단위 테스트
│   │   ├── services/
│   │   │   ├── test_load_calculator.py
│   │   │   ├── test_removal_analyzer.py
│   │   │   ├── test_statistical_analyzer.py
│   │   │   ├── test_process_simulator.py
│   │   │   ├── test_anomaly_detector.py
│   │   │   ├── test_decision_engine.py
│   │   │   └── test_control_optimizer.py
│   │   ├── ml/
│   │   │   ├── test_stage_predictor.py
│   │   │   ├── test_anomaly_classifier.py
│   │   │   └── test_model_loader.py
│   │   └── utils/
│   │       ├── test_validators.py
│   │       └── test_calculators.py
│   │
│   ├── fixtures/                 # 테스트 데이터 및 픽스처
│   │   ├── sample_water_quality.json
│   │   ├── sample_process_data.json
│   │   └── mock_models/
│   │
│   └── conftest.py               # pytest 설정 및 공통 픽스처
│
├── scripts/                      # 유틸리티 스크립트
│   ├── init_db.py                # 데이터베이스 초기화
│   ├── train_models.py           # 모델 학습 실행
│   ├── export_models.py          # 모델 내보내기
│   └── seed_data.py              # 샘플 데이터 생성
│
├── models/                       # 훈련된 모델 저장소
│   ├── stage_models/
│   ├── anomaly_models/
│   └── control_models/
│
├── docker/
│   ├── Dockerfile                # FastAPI 애플리케이션 이미지
│   ├── Dockerfile.train          # 모델 학습용 이미지 (선택)
│   └── docker-compose.yml        # 전체 스택 오케스트레이션
│
├── alembic.ini                   # Alembic 설정
├── pyproject.toml                # Python 프로젝트 설정 및 의존성
├── pytest.ini                    # pytest 설정
├── .env.example                  # 환경 변수 예시
└── README.md                     # 프로젝트 개요 및 설정 가이드
```

**Structure Decision**:
웹 API 백엔드 전용 프로젝트로 구조화. FastAPI 베스트 프랙티스를 따라 계층별로 명확히 분리:
- `api/`: 라우팅 및 엔드포인트 (프레젠테이션 레이어)
- `services/`: 비즈니스 로직 및 도메인 레이어
- `models/`: 데이터베이스 ORM 모델
- `schemas/`: API 입출력 검증 스키마
- `ml/`: AI/ML 모델 관리 (모델 정의, 학습, 추론)
- `db/`: 데이터베이스 연결 및 마이그레이션

6개 주요 기능 영역별로 라우터와 서비스를 분리하여 독립적 개발 및 테스트 가능.

## Complexity Tracking

> **Constitution Check에서 정당화가 필요한 위반 사항이 있는 경우에만 작성**

현재 복잡도 위반 사항 없음. 단일 백엔드 프로젝트로 적절히 구조화됨.

---

## Phase 0: 개요 및 연구

### 연구 목표

Technical Context에서 식별된 미결정 사항 및 기술 선택에 대한 조사 수행:

1. **PyTorch 모델 서빙 최적화**: FastAPI에서 PyTorch 모델을 효율적으로 로드하고 추론하는 베스트 프랙티스
2. **비동기 처리 패턴**: FastAPI + SQLAlchemy 비동기 패턴, ML 추론 비동기 처리
3. **모델 버전 관리**: 훈련된 PyTorch 모델의 버전 관리 및 배포 전략
4. **공정 시뮬레이션 아키텍처**: 9단계 순차 처리 공정의 효율적인 파이프라인 설계
5. **이상 탐지 임계값 관리**: 통계 기반 동적 임계값 설정 및 저장 방법
6. **MariaDB 스키마 설계**: 시계열 수질 데이터, 계산 결과, 이력 데이터의 효율적 저장
7. **Docker 멀티스테이지 빌드**: PyTorch + FastAPI 최적화된 컨테이너 이미지 구축
8. **성능 모니터링**: API 응답 시간 및 모델 추론 성능 추적 방법

### 연구 에이전트 할당

각 연구 주제별로 Task 에이전트를 실행하여 조사 결과를 `research.md`에 통합 예정.

**Output**: research.md (모든 기술 결정 사항 문서화)

**상태**: ✅ 완료 - research.md 생성됨

---

## ML 모델 구현 전략

### 개요

하수처리장 AI API 서비스는 PyTorch 기반의 다양한 ML 모델을 활용합니다. 대부분의 모델은 전통적인 ML 알고리즘(XGBoost, LightGBM)과 경량 신경망의 혼합으로 구성되며, 각 처리 공정 단계별로 특화된 모델을 사용합니다.

### 모델 아키텍처 전략

#### 1. 공정 단계별 예측 모델 (Stage-Specific Predictors)

**모델 유형**: Multi-Output Regression (XGBoost, PyTorch MLP)

**현재 구현 상태 (기존 코드 기반)**:
- **유입수 (Influent)**: XGBoost Multi-Output (54개 피처 → BOD, TN, TP 예측)
- **이차침전지 (Secondary Clarifier)**: XGBoost Regression

**확장 필요 공정**:
- 일차침전지 (Primary Clarifier)
- 혐기조 (Anaerobic Tank)
- 무산소조 (Anoxic Tank)
- 호기조 (Aerobic Tank)
- 분리막 (Membrane)
- 총인처리 (Phosphorus Treatment)
- 방류수 (Effluent)

**모델 구조 (PyTorch MLP 예시)**:
```python
class StagePredictor(nn.Module):
    """
    공정 단계별 수질 예측 모델

    입력: 이전 단계 수질 + 현재 단계 운전 파라미터
    출력: 현재 단계 출구 수질
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 각 공정별 모델 인스턴스
models = {
    "influent": StagePredictor(input_dim=54, hidden_dims=[128, 64, 32], output_dim=3),
    "primary_clarifier": StagePredictor(input_dim=30, hidden_dims=[64, 32], output_dim=5),
    "anaerobic": StagePredictor(input_dim=25, hidden_dims=[64, 32], output_dim=4),
    "anoxic": StagePredictor(input_dim=28, hidden_dims=[64, 32], output_dim=5),
    "aerobic": StagePredictor(input_dim=35, hidden_dims=[128, 64], output_dim=8),
    "secondary_clarifier": StagePredictor(input_dim=40, hidden_dims=[64, 32], output_dim=6),
    # ...
}
```

**학습 전략**:
1. **개별 학습**: 각 공정 모델을 독립적으로 학습
2. **순차 Fine-tuning**: 전체 파이프라인 시뮬레이션 결과로 end-to-end 미세조정
3. **Transfer Learning**: 유사 공정간 가중치 공유 (예: 혐기조 → 무산소조)

#### 2. 이상 탐지 모델 (Anomaly Detection)

**모델 유형**: Classification + Regression 혼합

**아키텍처 선택**:
- **통계 기반 임계값** (Phase 1): 빠른 구현, 해석 용이
- **Autoencoder** (Phase 2): 비선형 패턴 학습
- **Isolation Forest** (Phase 2): 다변량 이상치 탐지

**PyTorch Autoencoder 예시**:
```python
class AnomalyDetector(nn.Module):
    """
    Autoencoder 기반 이상 탐지 모델
    재구성 오차가 큰 샘플을 이상으로 판단
    """
    def __init__(self, input_dim, encoding_dim=16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def detect_anomaly(self, x, threshold):
        """재구성 오차로 이상 여부 판단"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = ((x - reconstructed) ** 2).mean(dim=1)
            is_anomaly = mse > threshold
            return is_anomaly, mse
```

**학습 목표**:
- Precision ≥ 95%, Recall ≥ 95%, F1-Score ≥ 0.85 (FR-023)

#### 3. 제어 파라미터 최적화 모델 (Control Optimization)

**모델 유형**: Regression (PyTorch MLP, XGBoost)

**최적화 대상** (FR-032~036):
1. 송풍량 (Aeration Rate)
2. 외부탄소원 투입량 (External Carbon Dosing)
3. 응집제 투입량 (Coagulant Dosing)
4. 역세척 주기 (Backwash Timing)
5. 슬러지 인발량 (Sludge Withdrawal)

**강화학습 기반 최적화 (Advanced, Optional)**:
```python
class ControlOptimizer(nn.Module):
    """
    현재 상태 → 최적 제어 파라미터 매핑

    입력: [현재 수질, 운전 파라미터, 목표 수질]
    출력: [최적 DO 설정값, 송풍량, 탄소원 투입량, ...]
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 출력 범위 제한 (-1, 1)
        )

    def forward(self, state):
        return self.policy_net(state)
```

### 모델 학습 파이프라인

#### 1. 데이터 전처리

```python
class WastewaterDataPreprocessor:
    """하수처리장 데이터 전처리 파이프라인"""

    def __init__(self, feature_cols, target_cols):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit_transform(self, df):
        """학습 데이터 전처리"""
        # 이상치 처리 (IQR 방법)
        df_clean = self.remove_outliers(df)

        # 결측치 처리 (시계열 보간)
        df_filled = self.fill_missing(df_clean)

        # 피처 엔지니어링
        df_features = self.engineer_features(df_filled)

        # 스케일링
        X = self.scaler_X.fit_transform(df_features[self.feature_cols])
        y = self.scaler_y.fit_transform(df_features[self.target_cols])

        return X, y

    def remove_outliers(self, df, factor=1.5):
        """IQR 기반 이상치 제거"""
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        return df[(df >= lower_bound) & (df <= upper_bound)].dropna()

    def fill_missing(self, df):
        """시계열 보간"""
        return df.interpolate(method='time', limit=3)

    def engineer_features(self, df):
        """도메인 특화 피처 생성"""
        df = df.copy()

        # 부하 계산
        df['BOD_load'] = df['유입유량'] * df['유입BOD'] / 1000
        df['TN_load'] = df['유입유량'] * df['유입TN'] / 1000

        # 비율 계산
        df['F_M_ratio'] = df['BOD_load'] / (df['MLSS'] * df['반응조용량'])
        df['C_N_ratio'] = df['유입COD'] / df['유입TN']

        # 시간 특성
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df
```

#### 2. 모델 학습

```python
class ModelTrainer:
    """모델 학습 클래스"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        patience=10
    ):
        """모델 학습 (Early Stopping)"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        return self.history
```

#### 3. 모델 평가

```python
class ModelEvaluator:
    """모델 성능 평가"""

    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """회귀 모델 평가 지표"""
        from sklearn.metrics import r2_score, mean_absolute_percentage_error

        metrics = {}

        # R² Score
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100

        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))

        return metrics

    @staticmethod
    def evaluate_anomaly_detection(y_true, y_pred):
        """이상 탐지 모델 평가 (FR-023 요구사항)"""
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # FR-023 요구사항 검증
        meets_requirements = (
            precision >= 0.95 and
            recall >= 0.95 and
            f1 >= 0.85
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'meets_requirements': meets_requirements,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'report': classification_report(y_true, y_pred)
        }
```

### 모델 배포 및 서빙

#### 1. 모델 저장 형식

```python
class ModelArtifact:
    """모델 아티팩트 관리"""

    @staticmethod
    def save(model, metadata, save_dir):
        """모델 및 메타데이터 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = metadata.get('version', '1.0.0')

        # 모델 저장
        model_path = save_dir / f"model_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.config,
            'version': version,
            'timestamp': timestamp,
        }, model_path)

        # 전처리 객체 저장
        scaler_path = save_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(metadata['scaler'], scaler_path)

        # 메타데이터 저장
        metadata_path = save_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_path, scaler_path, metadata_path
```

#### 2. 모델 로딩 및 캐싱

```python
class ModelLoader:
    """모델 로드 및 캐싱 관리"""

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self._cache = {}

    def load_model(self, model_name, version='latest'):
        """모델 로드 (캐싱)"""
        cache_key = f"{model_name}_{version}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # 모델 경로 찾기
        model_path = self._find_model_path(model_name, version)

        # 모델 로드
        checkpoint = torch.load(model_path, map_location='cpu')

        model = self._create_model_instance(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 메모리 최적화
        for param in model.parameters():
            param.requires_grad = False

        # 캐싱
        self._cache[cache_key] = model

        return model
```

### 지속적 학습 (Continuous Training)

#### 재학습 트리거 조건

1. **성능 저하 감지**:
   - 최근 7일 예측 MAPE > 임계값 + 10%
   - R² Score < 임계값 - 0.05

2. **데이터 드리프트 감지**:
   - 입력 분포 변화 (KS-test p-value < 0.05)
   - 새로운 운전 조건 출현

3. **정기 재학습**:
   - 월 1회 scheduled 재학습
   - 새 데이터 1000개 이상 누적 시

#### 재학습 파이프라인

```python
class ContinuousTrainingPipeline:
    """지속적 학습 파이프라인"""

    async def check_retraining_needed(self, model_name):
        """재학습 필요 여부 확인"""
        # 최근 성능 메트릭 조회
        recent_metrics = await self.get_recent_metrics(model_name, days=7)

        # 성능 저하 확인
        if recent_metrics['mape'] > self.threshold_mape * 1.1:
            return True, "Performance degradation detected"

        # 데이터 드리프트 확인
        drift_detected = await self.detect_data_drift(model_name)
        if drift_detected:
            return True, "Data drift detected"

        # 정기 재학습 확인
        last_training = await self.get_last_training_date(model_name)
        if (datetime.now() - last_training).days > 30:
            return True, "Scheduled retraining"

        return False, "No retraining needed"

    async def retrain_model(self, model_name):
        """모델 재학습 실행"""
        # 1. 최신 데이터 로드
        new_data = await self.fetch_training_data(days=90)

        # 2. 모델 학습
        trainer = ModelTrainer(model=self.models[model_name])
        history = trainer.train(new_data['train'], new_data['val'])

        # 3. 성능 평가
        metrics = self.evaluate(new_data['test'])

        # 4. 기존 모델과 비교
        if metrics['r2'] > self.current_metrics['r2']:
            # 새 모델 배포
            await self.deploy_new_model(model_name, metrics)
        else:
            # 기존 모델 유지
            print("New model performance not improved, keeping current model")
```

### 모델 성능 목표

| 모델 유형 | 평가 지표 | 목표값 |
|----------|---------|--------|
| 공정 예측 (Regression) | R² Score | ≥ 0.85 |
|  | MAPE | < 10% |
|  | 추론 시간 | < 20ms |
| 이상 탐지 (Classification) | Precision | ≥ 95% |
|  | Recall | ≥ 95% |
|  | F1-Score | ≥ 0.85 |
| 제어 최적화 (Regression) | 목표 달성률 | ≥ 85% |
|  | 안전 범위 준수율 | 100% |

### 모델 관리 도구

**MLflow 통합 (Optional)**:
```python
import mlflow

class MLflowTracker:
    """MLflow 실험 추적"""

    def log_training(self, model_name, params, metrics, model):
        """학습 결과 로깅"""
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # 파라미터 로깅
            mlflow.log_params(params)

            # 메트릭 로깅
            mlflow.log_metrics(metrics)

            # 모델 저장
            mlflow.pytorch.log_model(model, "model")

            # 아티팩트 저장
            mlflow.log_artifact("training_plot.png")
```

**모델 모니터링 대시보드**:
- 실시간 추론 성능 (latency, throughput)
- 예측 정확도 추이
- 데이터 드리프트 탐지
- 모델 버전별 A/B 테스트 결과

---

## Phase 1: 설계 및 계약

**전제 조건**: `research.md` 완료

### 1.1 데이터 모델 추출

Feature spec의 Key Entities를 기반으로 `data-model.md` 생성:

**엔티티**:
- Treatment Process (처리 공정)
- Water Quality Measurement (수질 측정)
- Load Calculation (부하 계산)
- Removal Efficiency (제거율)
- Process Indicator (공정 지표)
- Anomaly Event (이상 이벤트)
- Decision Recommendation (의사결정 추천)
- Control Setpoint (제어 설정값)
- Simulation Result (시뮬레이션 결과)

각 엔티티에 대해:
- 필드 정의 (타입, 제약조건)
- 관계 (FK, 1:N, N:M)
- 유효성 검증 규칙
- 상태 전이 (해당되는 경우)

### 1.2 API 계약 생성

Functional Requirements를 기반으로 OpenAPI 스키마 생성:

**API 엔드포인트 그룹**:
1. **Load Analysis API** (FR-003, FR-004):
   - `POST /api/v1/load-analysis/calculate` - 부하 계산
   - `POST /api/v1/load-analysis/removal-efficiency` - 제거율 분석

2. **Process Analysis API** (FR-011, FR-012):
   - `POST /api/v1/process-analysis/statistical` - 통계 분석
   - `POST /api/v1/process-analysis/clustering` - 군집 분석
   - `POST /api/v1/process-analysis/correlation` - 상관 분석
   - `GET /api/v1/process-analysis/operational-map` - 운전지도 데이터

3. **Simulation API** (FR-018):
   - `POST /api/v1/simulation/full-chain` - 전체 공정 시뮬레이션
   - `POST /api/v1/simulation/stage/{stage_name}` - 단계별 시뮬레이션

4. **Anomaly Detection API** (FR-024):
   - `POST /api/v1/anomaly/detect` - 이상 탐지
   - `GET /api/v1/anomaly/events` - 이상 이벤트 조회
   - `GET /api/v1/anomaly/performance` - 탐지 성능 메트릭

5. **Decision Support API** (FR-031):
   - `POST /api/v1/decision/recommend` - 운영 파라미터 추천
   - `POST /api/v1/decision/simulate-outcome` - 추천 결과 시뮬레이션
   - `GET /api/v1/decision/history` - 의사결정 이력

6. **Control Optimization API** (FR-037):
   - `POST /api/v1/control/optimize/aeration` - 송풍 최적화
   - `POST /api/v1/control/optimize/carbon-dosing` - 탄소원 투입 최적화
   - `POST /api/v1/control/optimize/coagulant` - 응집제 투입 최적화
   - `POST /api/v1/control/optimize/backwash` - 역세척 주기 최적화
   - `POST /api/v1/control/optimize/sludge-withdrawal` - 슬러지 인발 최적화

각 엔드포인트에 대해 OpenAPI 스키마 정의 (`/contracts/` 디렉토리).

### 1.3 Quickstart 문서 생성

`quickstart.md` 생성:
- 로컬 개발 환경 설정
- Docker Compose로 전체 스택 실행
- API 호출 예제 (curl, Python requests)
- 샘플 데이터로 테스트 시나리오 실행

### 1.4 Agent Context 업데이트

`.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude` 실행하여 현재 기술 스택을 에이전트 컨텍스트에 추가.

**Output**: data-model.md, /contracts/*, quickstart.md, 업데이트된 에이전트 컨텍스트

---

## Phase 2: 작업 분해

**Note**: 이 단계는 `/speckit.tasks` 명령으로 별도 실행. `/speckit.plan`에서는 생성하지 않음.

---

## 다음 단계

1. Phase 0 연구 진행 (research.md 생성)
2. Phase 1 설계 작업 (data-model.md, contracts/, quickstart.md 생성)
3. Constitution Check 재평가
4. `/speckit.tasks` 실행하여 구현 작업 분해

**참고**: 한국어로 작성된 구현 계획이 완료되었습니다. Phase 0 연구를 진행하여 기술적 결정 사항을 문서화하겠습니다.
