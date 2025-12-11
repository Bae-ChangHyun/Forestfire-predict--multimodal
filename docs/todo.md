# TODO - 프로젝트 분석 및 개선

## 프로젝트 개요
- **목적**: 강원도 산불 발생 확률 예측 (멀티모달 딥러닝)
- **데이터**:
  - Tabular: 기상 데이터 (기온, 습도, 풍속, 강수량)
  - Image: 지형/인적 데이터 (고도, 경사도, NDVI, 인구밀도, 토지이용도)
- **기술**: TensorFlow 2.6.0, Python 3.9.0, GDAL, rasterio
- **기간**: 2023.05 ~ 2023.11

---

## 접근방식 재검토 및 대안 제시

### 현재 접근방식 분석

**핵심 아이디어**: 수치값(tabular)만으로 예측이 어려워 **공간적/시각적 패턴**을 활용
- 지형 데이터 (고도, 경사도, NDVI): 공간적 특성
- 인적 데이터 (인구밀도, 토지이용): 공간적 분포
- 기상 데이터 (기온, 습도, 풍속, 강수): 시간적 변화

**현재 구현 (Multi-branch CNN)**:
```
5개 이미지 입력 → 각각 독립 CNN → Concatenate → Dense → 확률
기상 데이터 → Dense
```

**문제점**:
1. 각 이미지를 독립적으로 처리 → 모달 간 상호작용 학습 안 됨
2. 공간적 관계 무시 (인접 격자 간 영향)
3. 단일 시점만 사용 → 시계열 패턴 활용 안 함
4. 278x400 전체 격자를 예측하는데, 학습은 점(point) 기반

---

### 대안 접근방식 제안

#### 방식 1: **Graph Neural Network (GNN) 기반** ⭐ 추천

**핵심 아이디어**: 강원도를 그래프로 모델링
- **노드**: 각 격자점 (278 × 400 = 111,200개)
- **엣지**: 인접 격자 간 연결 (4-neighbor 또는 8-neighbor)
- **노드 특징**: [기온, 습도, 풍속, 강수, 고도, 경사도, NDVI, 인구밀도, 토지이용]
- **목표**: 각 노드의 산불 발생 확률 예측

**장점**:
- ✅ 공간적 관계 명시적 학습 (인접 지역 영향)
- ✅ 산불의 전파 패턴 모델링 가능
- ✅ 이미지를 픽셀이 아닌 구조화된 데이터로 처리
- ✅ Interpretable (어떤 이웃이 영향 주는지 분석 가능)
- ✅ 확장 가능 (시계열 GNN으로 발전 가능)

**단점**:
- ⚠️ 그래프 구성 오버헤드
- ⚠️ 전체 격자 학습시 메모리 사용량 큼 → Mini-batch 필요

**구현 방안**:
```python
# PyTorch Geometric 사용
class WildfireGNN(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(9, 64)  # 9개 feature
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 1)  # 확률 예측

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return torch.sigmoid(self.fc(x))
```

---

#### 방식 2: **Spatial-Temporal Transformer**

**핵심 아이디어**: 격자를 sequence로 보고 Transformer 적용
- 각 격자를 패치로 나누기 (예: 16x16 패치)
- 각 패치의 특징 추출 후 positional encoding
- Self-attention으로 전역 패턴 학습

**장점**:
- ✅ 장거리 의존성 학습 (멀리 떨어진 지역 영향)
- ✅ 최신 아키텍처 (SOTA 성능)
- ✅ 시계열 확장 용이 (시간축 추가)

**단점**:
- ⚠️ 많은 데이터 필요 (현재 1,368개는 부족할 수 있음)
- ⚠️ 계산 비용 높음
- ⚠️ 작은 패턴 놓칠 수 있음

---

#### 방식 3: **Tabular-First Hybrid** (효율성 중시)

**핵심 아이디어**: 기상 데이터 메인 + 공간 정보는 임베딩으로
- XGBoost/LightGBM으로 tabular 데이터 처리
- 공간 정보 (좌표, 지형)를 feature로 추가
- 이미지에서 CNN으로 추출한 특징을 추가 feature로 사용

**장점**:
- ✅ 빠른 학습
- ✅ 적은 데이터로도 잘 작동
- ✅ Feature importance 분석 쉬움
- ✅ 현실적으로 가장 빠르게 개선 가능

**단점**:
- ⚠️ 복잡한 공간 패턴 학습 제한적
- ⚠️ 이미지의 장점 완전히 활용 못 함

**구현 방안**:
```python
# 1. CNN으로 공간 특징 추출
spatial_features = cnn_extract(images)  # (n, 128)

# 2. Tabular 데이터와 결합
X = [기온, 습도, 풍속, 강수, 위도, 경도, spatial_features...]

# 3. XGBoost로 학습
model = xgb.XGBClassifier()
model.fit(X, y)
```

---

#### 방식 4: **Multi-Scale CNN with Attention** (현재 방식 개선)

**핵심 아이디어**: 현재 CNN 유지하되 크게 개선
- Feature Pyramid Network로 다양한 스케일 포착
- Cross-modal Attention으로 모달 간 상호작용
- Spatial Attention으로 중요 영역 강조

**장점**:
- ✅ 현재 구조 활용 (migration 쉬움)
- ✅ 검증된 방법
- ✅ 구현 난이도 낮음

**단점**:
- ⚠️ 여전히 독립적 처리의 한계
- ⚠️ 공간적 관계 약함

**구현 방안**:
```python
# Feature Pyramid Network
def fpn_branch(input):
    # Multi-scale features
    f1 = conv_block(input, scale=1)  # 원본
    f2 = conv_block(input, scale=2)  # 1/2
    f3 = conv_block(input, scale=4)  # 1/4
    return concatenate([f1, upsample(f2), upsample(f3)])

# Cross-modal Attention
def cross_attention(feature_a, feature_b):
    attention_weights = softmax(dot(feature_a, feature_b.T))
    return matmul(attention_weights, feature_b)

# 각 모달 처리 후 attention
climate_feat = dense_layers(climate)
height_feat = fpn_branch(height)
ndvi_feat = fpn_branch(ndvi)
...

# Cross attention
enhanced_height = cross_attention(height_feat, climate_feat)
enhanced_ndvi = cross_attention(ndvi_feat, climate_feat)
...
```

---

#### 방식 5: **U-Net 기반 Segmentation 접근**

**핵심 아이디어**: 산불 예측을 semantic segmentation 문제로
- 입력: 9-channel 이미지 (모든 feature를 channel로)
- 출력: 278×400 확률 맵
- U-Net으로 end-to-end 학습

**장점**:
- ✅ Pixel-wise 예측에 특화
- ✅ Skip connection으로 low-level + high-level 정보 결합
- ✅ 이미지 분야에서 검증된 구조

**단점**:
- ⚠️ 기상 데이터 통합이 어색함
- ⚠️ 데이터 증강 필요

---

### 권장 접근방식 (우선순위)

#### 🥇 **1순위: Graph Neural Network (GNN)**

**이유**:
- 산불의 본질적 특성과 잘 맞음 (공간적 전파, 인접 영향)
- 이미지를 "공간 구조"로 재해석 (픽셀 → 격자점)
- 확장성 높음 (시계열, 동적 예측 가능)
- 모든 feature를 통합하기 쉬움

**Phase 1 구현**:
1. 격자 그래프 구성 (4-neighbor)
2. 노드 특징: 9개 feature
3. 간단한 GCN 3-layer
4. Mini-batch GraphSAINT 샘플링

**Phase 2 확장**:
- Temporal GNN (시계열 추가)
- Attention 메커니즘
- Edge weight 학습 (거리 기반)

---

#### 🥈 **2순위: Tabular-First Hybrid (빠른 개선)**

**이유**:
- 가장 빠르게 성능 개선 가능
- 해석 가능성 높음
- 실무적으로 유용

**구현 방안**:
1. CNN으로 각 이미지에서 특징 추출 (pretrained ResNet)
2. Tabular 데이터 + 공간 좌표 + CNN 특징 결합
3. LightGBM으로 학습

---

#### 🥉 **3순위: Multi-Scale CNN with Attention (안전한 개선)**

**이유**:
- 현재 코드베이스 활용
- 검증된 방법
- Migration risk 낮음

---

### 최종 제안: Hybrid GNN-CNN

**핵심 아이디어**: GNN과 CNN의 장점 결합

```
[이미지 데이터] → CNN → Spatial Features (128-dim)
[기상 데이터] → 격자별로 할당

↓

각 격자 노드 = [기상(4) + CNN특징(128)] = 132-dim

↓

GNN (3-layer GCN)
- Layer 1: 132 → 256
- Layer 2: 256 → 128
- Layer 3: 128 → 64

↓

MLP Head: 64 → 1 (확률)
```

**장점**:
- ✅ CNN으로 이미지의 지역적 패턴 포착
- ✅ GNN으로 공간적 관계 학습
- ✅ 두 방식의 장점 결합
- ✅ 점진적 구현 가능 (먼저 CNN 개선 → GNN 추가)

---

## 발견된 개선점

### 1. 코드 구조 및 설계 (우선순위: 높음)

#### 1.1 모듈 간 의존성 문제
**문제**:
- `train_model.py:35-36`에서 subprocess로 다른 스크립트 실행
- 에러 핸들링 어려움, 비효율적

**개선 방안**:
- 함수 import 방식으로 변경
- 각 모듈을 독립적으로 실행 가능하도록 구조화

#### 1.2 전역 변수 남용
**문제**:
- `train_model.py:156-161`의 Input 레이어들이 전역 변수
- 코드 재사용성 및 테스트 어려움

**개선 방안**:
- 모델 생성 함수 내부로 이동
- 설정을 config 객체로 관리

#### 1.3 하드코딩된 값
**문제**:
- 격자 크기 (278, 400), Magic numbers (624, 32767, -9999) 등
- 유지보수 어려움

**개선 방안**:
- 상수를 config 파일로 분리
- Enum 또는 NamedTuple 사용

---

### 2. 데이터 처리 (우선순위: 높음)

#### 2.1 결측치 처리 불일치
**문제**:
- `tabular_dataset.py:356`에서 32767을 -9999로 대체
- 모델이 -9999를 정상 값으로 학습할 가능성

**개선 방안**:
- 결측치를 명시적으로 처리 (masking 또는 제거)
- NaN을 적절한 값으로 대체하거나 별도 플래그 생성

#### 2.2 데이터 불균형
**문제**:
- 산불(target=1) vs 비산불(target=0) 비율 불명확
- `make_random_dataset`의 샘플링 전략이 단순

**개선 방안**:
- 클래스 비율 분석 및 최적화
- SMOTE 또는 클래스 가중치 적용
- Stratified sampling 사용

#### 2.3 데이터 정규화 불일치
**문제**:
- 이미지: 0-1 정규화 (`image_dataset.py:148`)
- 기상 데이터: 정규화 없음
- 스케일 차이로 학습 불균형

**개선 방안**:
- 모든 입력에 StandardScaler 또는 MinMaxScaler 적용
- 정규화 파라미터 저장 (inference 시 재사용)

---

### 3. 모델 아키텍처 (우선순위: 중간)

#### 3.1 단순한 모델 구조
**문제**:
- CNN이 너무 단순 (4층 반복)
- Batch Normalization, Dropout 등 정규화 없음
- ResNet 블록 구현했으나 미사용 (`train_model.py:50-74`)

**개선 방안**:
- ResNet 또는 EfficientNet 백본 사용
- Batch Normalization, Dropout 추가
- 모델 depth/width 실험

#### 3.2 멀티모달 융합 방식
**문제**:
- 단순 concatenation만 사용 (`train_model.py:114`)
- 모달 간 상호작용 학습 부족

**개선 방안**:
- Cross-attention 메커니즘 추가
- Feature fusion layer (Bilinear pooling, Tucker decomposition 등)
- Late fusion vs Early fusion 비교 실험

#### 3.3 조기 종료 조건
**문제**:
- Train accuracy 0.95로 조기 종료 (`train_model.py:140-142`)
- Validation set 없어 과적합 감지 불가
- 메시지 오류 (코드: 0.95, 메시지: 0.9)

**개선 방안**:
- Validation accuracy 기반 조기 종료
- EarlyStopping, ReduceLROnPlateau 콜백 추가
- ModelCheckpoint로 best model 저장

---

### 4. 학습 전략 (우선순위: 높음)

#### 4.1 데이터 분리 문제
**문제**:
- Train/Validation/Test 분리 없음
- 과적합 감지 불가능

**개선 방안**:
- Train/Validation/Test를 7:2:1 또는 8:1:1로 분리
- Stratified split으로 클래스 비율 유지
- K-fold cross-validation 고려

#### 4.2 하이퍼파라미터
**문제**:
- Learning rate (0.0001), batch size (128) 고정
- 최적화 없음

**개선 방안**:
- Learning rate scheduler 추가
- Optuna, Ray Tune 등으로 하이퍼파라미터 튜닝
- Batch size, optimizer 실험

#### 4.3 손실 함수
**문제**:
- Binary crossentropy만 사용
- 클래스 불균형 미고려

**개선 방안**:
- Focal Loss 또는 Weighted BCE 사용
- Class weights 계산 및 적용

---

### 5. 코드 품질 (우선순위: 중간)

#### 5.1 에러 처리
**문제**:
- Bare `except:` 사용 (`tabular_dataset.py:70, 100, 110`)
- 에러 메시지 불명확

**개선 방안**:
- 구체적인 Exception 타입 지정
- 로깅을 통한 에러 추적
- 의미 있는 에러 메시지

#### 5.2 코드 중복
**문제**:
- CNN 레이어 5번 반복 (`train_model.py:107-111`)
- 보간 코드 중복

**개선 방안**:
- 함수화 및 루프 사용
- DRY 원칙 적용

#### 5.3 주석 및 문서화
**문제**:
- 영어/한국어 혼용
- Docstring 부족

**개선 방안**:
- 일관된 언어 사용 (한국어 또는 영어)
- Google/NumPy 스타일 docstring 추가
- README 개선

#### 5.4 사용하지 않는 코드
**문제**:
- IPython import (`train_model.py:20-21`)
- 미사용 함수들 (residual_block, res_layers, 등)

**개선 방안**:
- Dead code 제거
- 코드 정리

---

### 6. 의존성 및 환경 (우선순위: 높음)

#### 6.1 구버전 라이브러리
**문제**:
- TensorFlow 2.6.0 (보안 취약점)
- Python 3.9.0

**개선 방안**:
- TensorFlow 2.15+ 업그레이드
- Python 3.11+ 고려
- 호환성 테스트

#### 6.2 플랫폼 호환성
**문제**:
- pywin32 (Windows 전용)
- 크로스 플랫폼 지원 부족

**개선 방안**:
- Optional dependency로 변경
- Docker 컨테이너화

#### 6.3 GDAL 설치
**문제**:
- `requirements.txt:38`에 로컬 경로
- 재현성 문제

**개선 방안**:
- conda 환경 사용 권장 (gdal 설치 용이)
- 또는 Docker 이미지 제공

---

### 7. API 및 외부 서비스 (우선순위: 낮음)

#### 7.1 재시도 로직
**문제**:
- Exponential backoff 없음
- 짧은 sleep time (2-5초)

**개선 방안**:
- tenacity 라이브러리 사용
- Exponential backoff with jitter

#### 7.2 API 키 관리
**문제**:
- 단순 검증만 (`load_variables.py:8-10`)

**개선 방안**:
- 환경 변수 검증 강화
- Secrets management 고려

---

### 8. 성능 및 효율성 (우선순위: 중간)

#### 8.1 메모리 관리
**문제**:
- 전체 데이터를 메모리에 로드
- OOM 위험

**개선 방안**:
- tf.data.Dataset 사용 (데이터 파이프라인)
- Generator 패턴 적용
- 배치 단위 로딩

#### 8.2 파일 I/O
**문제**:
- 반복적 파일 읽기/쓰기
- 캐싱 없음

**개선 방안**:
- TFRecord 형식 사용
- 중간 결과 캐싱
- Prefetch 및 parallel I/O

---

### 9. 테스트 및 검증 (우선순위: 중간)

#### 9.1 테스트 코드
**문제**:
- Unit test 없음
- Integration test 없음

**개선 방안**:
- pytest로 테스트 프레임워크 구축
- 데이터 처리 함수 테스트
- 모델 입출력 테스트

#### 9.2 모델 평가
**문제**:
- Accuracy만 사용
- 클래스별 성능 미분석

**개선 방안**:
- Precision, Recall, F1-Score, AUC-ROC 추가
- Confusion matrix 시각화
- 클래스별 성능 분석
- 공간적 오차 분석 (지역별 성능)

---

### 10. 프로젝트 구조 (우선순위: 중간)

#### 10.1 설정 파일
**문제**:
- config 파일 없음
- 모든 설정 하드코딩

**개선 방안**:
```
config/
  ├── config.yaml  # 전역 설정
  ├── model_config.yaml  # 모델 설정
  └── data_config.yaml  # 데이터 설정
```

#### 10.2 로깅
**문제**:
- print문만 사용
- 로그 저장/관리 없음

**개선 방안**:
- logging 모듈 사용
- 로그 레벨 관리 (DEBUG, INFO, WARNING, ERROR)
- 로그 파일 저장

#### 10.3 프로젝트 구조 개선
**현재 구조**:
```
code/
  ├── train_model.py
  ├── test_model.py
  ├── tabular_dataset.py
  ├── image_dataset.py
  ├── utils.py
  └── load_variables.py
```

**제안 구조**:
```
src/
  ├── config/
  │   └── config.yaml
  ├── data/
  │   ├── __init__.py
  │   ├── tabular_dataset.py
  │   ├── image_dataset.py
  │   └── data_loader.py
  ├── models/
  │   ├── __init__.py
  │   ├── base_model.py
  │   ├── cnn_branch.py
  │   └── multimodal_model.py
  ├── training/
  │   ├── __init__.py
  │   ├── trainer.py
  │   └── callbacks.py
  ├── inference/
  │   ├── __init__.py
  │   └── predictor.py
  ├── utils/
  │   ├── __init__.py
  │   ├── logger.py
  │   ├── metrics.py
  │   └── visualization.py
  └── main.py
tests/
  ├── test_data.py
  ├── test_model.py
  └── test_utils.py
scripts/
  ├── train.py
  ├── test.py
  └── preprocess.py
notebooks/
  └── exploration.ipynb
```

---

## 추가 개선 제안

### 11. 모델 개선 (고급)

#### 11.1 최신 아키텍처 적용
- Vision Transformer (ViT) 고려
- Swin Transformer
- ConvNeXt

#### 11.2 시공간 모델링
**문제**: 현재는 단일 시점만 사용
**개선**:
- LSTM/GRU로 시계열 패턴 학습
- 3D CNN (시공간 패턴)
- Temporal attention

#### 11.3 Explainable AI
- Grad-CAM으로 중요 영역 시각화
- SHAP values로 feature importance 분석
- Attention weights 시각화

### 12. 데이터 증강
- 이미지 증강 (rotation, flip, crop)
- Mixup, CutMix
- 기상 데이터 노이즈 추가

### 13. 앙상블
- 여러 모델 앙상블 (Voting, Stacking)
- K-fold 앙상블
- 다양한 아키텍처 조합

### 14. 배포 및 운영
- FastAPI로 서빙 API 구축
- Docker 컨테이너화
- CI/CD 파이프라인
- 모니터링 시스템

---

## 우선순위별 실행 계획 (개정)

### 🎯 선택된 접근방식: Hybrid GNN-CNN

**단계적 구현 전략**:
1. Phase 1-2: 기존 CNN 개선 (기반 다지기)
2. Phase 3: GNN 추가 (새로운 아키텍처)
3. Phase 4: Hybrid 통합 및 최적화

---

### Phase 1: 기반 구축 (1-2주)

#### 1.1 환경 및 프로젝트 구조
- [x] 접근방식 개선안 작성
- [ ] 브랜치 생성 (`refactor/improve-architecture`)
- [ ] uv 환경 설정 (venv 생성, 의존성 관리)
- [ ] 프로젝트 구조 개편 (`src/` 기반)
- [ ] Config 파일 시스템 구축 (YAML)
- [ ] 로깅 시스템 구축

#### 1.2 데이터 처리 개선
- [ ] 결측치 처리 통일 (masking 또는 명시적 제거)
- [ ] 데이터 정규화 통일 (StandardScaler)
- [ ] Train/Validation/Test 분리 (8:1:1, stratified)
- [ ] 클래스 불균형 분석 및 가중치 계산
- [ ] 데이터 로더 리팩토링 (tf.data.Dataset)

#### 1.3 모듈 의존성 개선
- [ ] `subprocess` 제거, import 방식으로 변경
- [ ] 전역 변수 제거, 함수형으로 리팩토링
- [ ] 하드코딩 값들을 config로 이동
- [ ] Dead code 제거

---

### Phase 2: CNN 모델 개선 (2-3주)

#### 2.1 기본 CNN 개선
- [ ] Batch Normalization 추가
- [ ] Dropout 추가 (overfitting 방지)
- [ ] ResNet 블록 활용
- [ ] Multi-scale Feature Extraction (FPN)

#### 2.2 멀티모달 융합 개선
- [ ] Cross-modal Attention 구현
- [ ] Feature concatenation → Learned fusion
- [ ] 각 모달별 중요도 학습 (gating mechanism)

#### 2.3 학습 전략 개선
- [ ] Focal Loss 또는 Weighted BCE
- [ ] Class weights 적용
- [ ] Learning rate scheduler
- [ ] Early stopping (validation 기반)
- [ ] Model checkpoint (best model 저장)

#### 2.4 평가 시스템
- [ ] Precision, Recall, F1-Score, AUC-ROC 추가
- [ ] Confusion matrix 시각화
- [ ] 클래스별 성능 분석
- [ ] 공간적 오차 분석 (지역별)

---

### Phase 3: GNN 아키텍처 구현 (3-4주)

#### 3.1 그래프 구성
- [ ] 격자 → 그래프 변환 (278×400 노드)
- [ ] 4-neighbor 엣지 구성
- [ ] 노드 특징 준비 (9-dim)
- [ ] PyTorch Geometric 환경 설정

#### 3.2 기본 GNN 모델
- [ ] GCN 3-layer 구현
- [ ] Mini-batch 샘플링 (GraphSAINT)
- [ ] 학습 파이프라인
- [ ] GNN 전용 평가 메트릭

#### 3.3 GNN 개선
- [ ] GAT (Graph Attention Network) 시도
- [ ] Edge weight 학습
- [ ] Skip connections
- [ ] Residual GNN

---

### Phase 4: Hybrid 통합 및 최적화 (2-3주)

#### 4.1 CNN-GNN Hybrid
- [ ] CNN으로 spatial features 추출
- [ ] GNN 노드 특징에 CNN features 통합
- [ ] End-to-end 학습
- [ ] Multi-task learning (보조 task 추가)

#### 4.2 최적화
- [ ] 하이퍼파라미터 튜닝 (Optuna)
- [ ] 데이터 증강 실험
- [ ] Ensemble (CNN + GNN + Hybrid)
- [ ] 추론 속도 최적화

#### 4.3 Interpretability
- [ ] Grad-CAM (CNN 중요 영역)
- [ ] Attention weights 시각화 (GNN)
- [ ] Feature importance 분석
- [ ] 예측 실패 케이스 분석

---

### Phase 5: 마무리 및 문서화 (1주)

#### 5.1 코드 정리
- [ ] 테스트 코드 작성 (pytest)
- [ ] Docstring 보완
- [ ] 코드 스타일 통일 (black, isort)
- [ ] Type hints 추가

#### 5.2 문서화
- [ ] README 업데이트
- [ ] 모델 아키텍처 다이어그램
- [ ] 실험 결과 정리
- [ ] 사용 가이드 작성

#### 5.3 실험 기록
- [ ] 각 접근방식별 성능 비교
- [ ] Ablation study
- [ ] 최종 모델 선정
- [ ] 개선 전후 비교

---

## 작업 환경 설정

### UV 사용
```bash
# 가상환경 생성
uv venv

# 활성화
source .venv/bin/activate

# 의존성 추가
uv add tensorflow>=2.15.0
uv add torch torchvision
uv add torch-geometric
uv add pandas numpy scikit-learn
uv add pyyaml python-dotenv
uv add matplotlib seaborn
uv add pytest black isort mypy
```

### 브랜치 전략
- `main`: 기존 코드 유지
- `refactor/improve-architecture`: 개선 작업
- Feature별 브랜치 (필요시)

### Commit Convention
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `chore`: 설정, 의존성 등
- `docs`: 문서
- `test`: 테스트 코드

---

## 메모
- 프로젝트가 2023년에 종료되었으나, 학술적/실무적 가치가 높음
- **접근방식 변경**: Multi-branch CNN → Hybrid GNN-CNN
- GNN이 산불의 공간적 전파 특성을 더 잘 모델링할 것으로 기대
- 단계적 구현으로 리스크 최소화
- TensorFlow → PyTorch 전환 (GNN 라이브러리 생태계)
- 배포/운영은 고려하지 않음 (개인 프로젝트)
