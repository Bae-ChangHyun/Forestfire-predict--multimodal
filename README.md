# Multimodal Forestfire Prediction
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBae-ChangHyun%2FForestfire-predict%2Fblob%2Fmain%2FREADME.md&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) <br>
<img alt="GitHub forks" src="https://img.shields.io/github/forks/Changroro/Forestfire-predict--multimodal">

개발기간: `2023.05 ~ 2023.11` (초기 버전) | `2024.12 ~` (리팩토링 및 개선)

Wiki: [Wiki](https://github.com/Changroro/Forestfire-predict--multimodal/wiki)

## 📌 Overview

대한민국 강원도의 산불 발생 확률(0~1)을 기상, 지형, 인적 데이터를 이용하여 예측하는 **멀티모달 딥러닝** 프로젝트입니다.

### 주요 특징

- **멀티모달 학습**: 기상 데이터(Tabular) + 지형/인적 데이터(Image)
- **다양한 모델 아키텍처**: CNN, GNN, Hybrid GNN-CNN
- **클래스 불균형 처리**: Focal Loss, Class Weighting
- **공간 관계 모델링**: Graph Neural Networks (GCN, GAT, GraphSAGE)
- **모듈화된 구조**: Config 기반, Factory 패턴

### 데이터

- **기상데이터**: 기온, 습도, 풍속, 강수량
- **지형데이터**: 고도, 경사도, NDVI
- **인적데이터**: 인구밀도, 토지이용도

프로젝트 회고: [블로그](https://changsroad.tistory.com/category/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/%EC%82%B0%EB%B6%88%20%EB%B0%9C%EC%83%9D%20%EC%98%88%EC%B8%A1)

## 🏗️ Project Structure

```
Multimodal-Forestfire-prediction/
├── src/
│   ├── config/              # Configuration files
│   │   ├── default.yaml     # Default configuration
│   │   └── config.py        # Config loader
│   ├── data/                # Data processing modules
│   │   ├── climate_data.py  # Climate data collection
│   │   ├── image_data.py    # Spatial data processing
│   │   ├── interpolation.py # IDW interpolation
│   │   ├── dataset.py       # Dataset classes
│   │   └── graph_data.py    # Graph construction for GNN
│   ├── models/              # Model architectures
│   │   ├── layers.py        # Custom layers (ResidualBlock, CBAM, etc.)
│   │   ├── cnn_branch.py    # CNN branches (Simple, ResNet, Attention)
│   │   ├── attention.py     # Attention mechanisms
│   │   ├── gnn.py           # GNN models (GCN, GAT, GraphSAGE)
│   │   ├── multimodal_model.py  # Multimodal CNN model
│   │   └── hybrid_model.py  # Hybrid GNN-CNN model
│   ├── training/            # Training utilities
│   │   ├── losses.py        # Loss functions (Focal Loss, etc.)
│   │   └── callbacks.py     # Training callbacks
│   └── utils/               # Utility modules
│       ├── constants.py     # Constants and enums
│       ├── preprocessing.py # Data preprocessing
│       ├── metrics.py       # Evaluation metrics
│       ├── visualization.py # Plotting functions
│       └── evaluator.py     # Model evaluation
├── scripts/                 # Executable scripts
│   ├── train.py            # Train multimodal CNN model
│   ├── train_gnn.py        # Train GNN/Hybrid model
│   ├── evaluate.py         # Evaluate trained model
│   └── predict.py          # Inference on new data
├── notebooks/              # Jupyter notebooks (TBD)
├── tests/                  # Unit tests (TBD)
├── code/                   # Legacy code (original version)
└── data/                   # Data directory
```

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/Changroro/Multimodal-Forestfire-prediction.git
cd Multimodal-Forestfire-prediction
```

### 2. Setup Environment (UV)

이 프로젝트는 `uv`를 사용하여 패키지를 관리합니다.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
uv sync
```

### 3. Download Data

데이터를 다운로드하고 `data/` 디렉토리에 배치합니다:

- [raw 파일](https://drive.google.com/file/d/1Kew7kQTDRqo_X_-T-rW06XjGvHvlBEMm/view?usp=drive_link)
- [ASOS 파일](https://drive.google.com/file/d/1KfERjVehpwHckMcY6gKZHB8tRyKIegVM/view?usp=drive_link)

⚠️ 데이터 용량이 크므로 충분한 저장 공간이 있는 드라이브에 설정하세요.

## 📖 Usage

### 1. Train Multimodal CNN Model (TensorFlow)

```bash
uv run python scripts/train.py \
    --config src/config/default.yaml \
    --data-dir ./data/processed \
    --output-dir ./outputs \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.0001 \
    --gpu 0
```

### 2. Train GNN/Hybrid Model (PyTorch)

```bash
# Hybrid GNN-CNN 모델
uv run python scripts/train_gnn.py \
    --config src/config/default.yaml \
    --data-dir ./data/processed \
    --model-type hybrid \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0001

# Pure GNN 모델
uv run python scripts/train_gnn.py \
    --model-type gnn \
    --data-dir ./data/processed
```

### 3. Evaluate Model

```bash
# 학습된 모델 평가
uv run python scripts/evaluate.py \
    --model-path ./outputs/20241211_120000/checkpoints/best_model.keras \
    --data-dir ./data/processed \
    --split test \
    --threshold 0.5
```

### 4. Inference on New Data

```bash
uv run python scripts/predict.py \
    --model-path ./outputs/20241211_120000/checkpoints/best_model.keras \
    --data-dir ./data/processed \
    --climate-file ./data/new_climate.csv \
    --output-file predictions.csv \
    --threshold 0.5 \
    --save-probabilities
```

## 🧠 Model Architectures

### 1. Multimodal CNN Model (TensorFlow)

- **5개 CNN Branches**: 각 spatial feature(고도, 경사도, NDVI, 토지이용, 인구밀도)마다 독립적인 CNN
- **Backbone Options**: Simple CNN, ResNet-style, Attention CNN (CBAM)
- **Climate Branch**: Dense layers for tabular climate data
- **Fusion**: Concatenation, Cross-modal Attention, Gated Fusion
- **Loss**: Focal Loss (α=0.25, γ=2.0) for class imbalance

### 2. Graph Neural Network Models (PyTorch)

- **그래프 구조**: 278×400 그리드 → 111,200 노드 그래프
- **Edge Connectivity**: 4-neighbor or 8-neighbor
- **GNN Variants**:
  - **GCN**: Graph Convolutional Network
  - **GAT**: Graph Attention Network (multi-head attention)
  - **GraphSAGE**: Inductive learning with neighborhood sampling
  - **Residual GCN**: GCN with skip connections

### 3. Hybrid GNN-CNN Model (PyTorch)

- **CNN**: Local spatial feature extraction
- **GNN**: Global spatial relationship modeling via graph structure
- **Fusion**: Gated fusion mechanism
- **End-to-End**: Joint training of CNN and GNN components

## 📊 Results

### Example: 2022.03.04 12:00 강원도 영월 산불

좌측: 국가산불위험예보시스템 | 우측: 프로젝트 산출물

<img src="https://github.com/Changroro/Forestfire-predict--multimodal/assets/48899047/a5336b36-6114-4da3-9316-51624fac2c19.png"  width="600" height="300"/>


## 📚 Tech Stack

### Deep Learning
- **TensorFlow / Keras**: Multimodal CNN models
- **PyTorch**: GNN and Hybrid models
- **PyTorch Geometric**: Graph Neural Networks

### Data Processing
- **GDAL / Rasterio**: Geospatial raster data processing
- **GeoPandas**: Spatial data manipulation
- **NumPy / Pandas**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and evaluation

### Development
- **UV**: Fast Python package manager
- **Git**: Version control
- **QGIS**: Geospatial visualization

### Key Libraries
```
tensorflow>=2.15.0
torch>=2.0.0
torch-geometric>=2.4.0
rasterio>=1.3.0
geopandas>=0.14.0
scikit-learn>=1.3.0
```

## 🎯 Key Improvements (v2.0)

### Architecture
- ✅ **Modular Structure**: Config-driven, factory pattern
- ✅ **Multiple Model Types**: CNN, GNN, Hybrid GNN-CNN
- ✅ **Attention Mechanisms**: CBAM, Cross-modal attention, Gated fusion
- ✅ **Residual Connections**: ResNet-style skip connections

### Data Processing
- ✅ **Unified Preprocessing**: MissingValueHandler, DataNormalizer
- ✅ **Stratified Splitting**: Maintain class ratios
- ✅ **Graph Construction**: Grid to graph conversion for GNN

### Training
- ✅ **Class Imbalance**: Focal Loss, Class weighting
- ✅ **Callbacks**: Early stopping, LR scheduling, Checkpointing
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR

### Evaluation
- ✅ **Visualization**: Confusion matrix, ROC/PR curves, Spatial error maps
- ✅ **Threshold Optimization**: Find optimal classification threshold
- ✅ **Model Comparison**: Compare multiple models

## 🔬 Future Work

- [ ] Temporal modeling (시계열 데이터 활용)
- [ ] Uncertainty quantification (예측 불확실성 정량화)
- [ ] Explainability (XAI: Grad-CAM, SHAP)
- [ ] Real-time inference optimization
- [ ] Web-based visualization interface

## 📝 Citation

If you use this code for your research, please cite:

```bibtex
@misc{forestfire-multimodal,
  author = {Bae, ChangHyun},
  title = {Multimodal Forestfire Prediction},
  year = {2023-2024},
  publisher = {GitHub},
  url = {https://github.com/Changroro/Multimodal-Forestfire-prediction}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contact

- Author: Bae ChangHyun
- Blog: [changsroad.tistory.com](https://changsroad.tistory.com)
- Issues: [GitHub Issues](https://github.com/Changroro/Multimodal-Forestfire-prediction/issues)
