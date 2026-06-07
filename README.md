# In Silico Biocompatible Peptide Reverse Engineering Pipeline

> AI-driven reverse engineering system that automatically designs peptide sequences minimizing Foreign Body Response (FBR) in implantable biomedical devices.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Final Results](#final-results)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Development History](#development-history)

---

## 🔬 Project Overview

생체재료 표면에 코팅되는 펩타이드 서열을 **역설계(Reverse Engineering)**하는 인실리코(In Silico) AI 파이프라인입니다.

기존 생체재료 개발은 실험적 시행착오(Trial & Error)에 의존하여 막대한 비용과 시간이 소요됩니다. 본 프로젝트는 **IEDB 면역 Epitope 데이터** 기반의 딥러닝 예측 모델과 **유전 알고리즘(Genetic Algorithm)** 최적화 루프를 결합하여, 대식세포의 이물 반응(FBR)을 최소화하는 펩타이드 서열을 자동으로 도출합니다.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Overall Pipeline                               │
│                                                                 │
│   [Pipeline A: FBR Predictor]                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  IEDB Dataset → BLOSUM62 Encoding → 1D-CNN + Bi-LSTM    │  │
│   │  → P_FBR Score (Inflammation Probability)               │  │
│   └──────────────────────────────────────────────────────────┘  │
│                          ↓ (Evaluator)                          │
│   [Pipeline B: GA-based Reverse Engineering]                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Initial Population (Warm-start: EVTELT motif)          │  │
│   │  → Multi-objective Cost Function                        │  │
│   │     (P_FBR + GRAVY + Charge + hCD47 Homology)          │  │
│   │  → Adaptive Mutation + Tournament Selection             │  │
│   │  → Optimal Peptide Sequence                             │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline A — FBR Predictor

- **목적:** 펩타이드 서열을 입력받아 대식세포의 염증 유발 확률(P_FBR) 예측
- **모델:** 1D-CNN (국소 모티프 탐지) + Bi-LSTM (양방향 서열 맥락 파악) 하이브리드
- **입력:** 8~15mer 펩타이드 서열 (BLOSUM62 인코딩, Pre-padding 적용)
- **출력:** P_FBR ∈ [0, 1] (0: 안전, 1: 염증 유발)
- **성능:** AUROC 0.8423 (Ablation Study 기준 1D-CNN 단독), 0.7235 (Hybrid)

### Pipeline B — GA Optimizer

- **목적:** Pipeline A를 심사위원(Evaluator)으로 삼아 최적 펩타이드 역설계
- **알고리즘:** Genetic Algorithm (50세대, Population=50)
- **비용 함수:** `Cost = w₁·P_FBR + w₂·|GRAVY| + w₃·|Charge| - w₄·hCD47_score`
- **핵심 메커니즘:**
  - **Warm-start:** hCD47 핵심 모티프(`EVTELT`) 기반 초기 인구 생성
  - **Adaptive Mutation:** 다양성 15% 미만 시 돌연변이율 동적 상승 (최대 50%)
  - **Smart Router:** 서열 길이에 따른 예측 엔진 동적 분기 (≤15: CNN Only, >15: Hybrid)
  - **Batch Inference:** 50개 개체군 텐서 일괄 처리로 연산 가속

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch |
| **Biological Encoding** | BLOSUM62 (substitution matrix) |
| **Optimization** | Genetic Algorithm (custom implementation) |
| **Bioinformatics** | BioPython, IEDB Dataset |
| **Structure Prediction** | ColabFold (AlphaFold2), SASA surrogate model |
| **Evaluation** | scikit-learn (AUROC, AUPRC, F1), K-Fold CV |
| **Explainability (XAI)** | PyTorch Forward Hook (CNN feature map + LSTM L2-Norm) |
| **Performance Profiling** | `time`, `psutil` |

---

## 🏆 Final Results

### Optimal Sequence

```
EVTELTLLTFHYKLR
```

### KPI Achievement (6/6 ✅)

| KPI | Target | Result | Status |
|---|---|---|---|
| P_FBR (Inflammation Prob.) | < 40% | ✅ Met | ✅ |
| GRAVY Score | ≈ 0 | ✅ Met | ✅ |
| Net Charge | -0.5 ~ 0.5 | ✅ Met | ✅ |
| hCD47 Homology | ≥ threshold | ✅ Met | ✅ |
| Computation Time | < 10 min | 327s (5m 27s) | ✅ |
| Peak Memory | < 500MB | ~170MB | ✅ |

### Best Model

- **Model:** Seq Hybrid (Pool)
- **Selection Basis:** `Total_Score = Mean_AUROC - W_STD·Std - W_TIME·Time - W_MEM·Memory`
- **Auto-selected from:** `benchmark_report.csv`

---

## 📁 Directory Structure

```
in-silico-peptide-reverse-engineering/
│
├── README.md                          # 프로젝트 소개 (이 파일)
│
├── src/                               # 소스 코드
│   ├── pipeline_b_only.py             # 메인 실행 파일 (Pipeline B: GA 역설계)
│   ├── 시스템_설계안_for_anyone.py     # 시스템 전체 설계안 (Pipeline A+B 통합)
│   └── 시스템설계안_중간영상_이후ver.ipynb  # 중간 이후 버전 노트북
│
├── notebooks/                         # 탐색/실험용 Jupyter Notebooks
│   ├── IEDB_데이터_추출.ipynb          # IEDB 데이터 전처리 파이프라인
│   └── 시스템_설계안_중간영상_Ver.ipynb # 중간 발표 버전 노트북
│
├── data/                              # 데이터 파일
│   ├── peptide_dataset_5_50.csv        # 전처리된 학습 데이터셋 (803,648 rows)
│   └── .gitignore                     # ⚠️ 대용량 원본 데이터는 .gitignore 처리
│
├── results/                           # 실험 결과
│   └── benchmark_report.csv           # Ablation Study 결과 (모델별 AUROC/메모리/시간)
│
└── docs/
    └── history/                       # 개발 과정 기록 (Gemini 대화 → MD 변환)
        ├── 작업과정_1.md
        ├── 작업과정_2.md
        ├── 작업과정_3.md
        ├── 작업과정_4.md
        ├── 작업과정_5.md
        ├── 작업과정_6.md
        ├── 작업과정_7.md
        ├── 작업과정_8.md
        └── 작업과정_9.md
```

> ⚠️ **`.gitignore` 처리 권장 파일:**
> - `data/` 하위 원본 대용량 CSV (IEDB raw data 등)
> - `results/` 하위 중간 체크포인트 `.pt` 파일
> - `__pycache__/`, `.ipynb_checkpoints/`

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch biopython scikit-learn pandas numpy psutil
```

### Run Pipeline B (GA-based Reverse Engineering)

```bash
# 1. 레포 클론
git clone https://github.com/YOUR_USERNAME/in-silico-peptide-reverse-engineering.git
cd in-silico-peptide-reverse-engineering

# 2. 데이터 확인 (data/ 디렉토리에 peptide_dataset_5_50.csv 위치)
ls data/

# 3. 메인 파이프라인 실행
python src/pipeline_b_only.py
```

### Expected Output

```
[GA] Generation 1/50 | Best Cost: X.XXX | P_FBR: XX.X% | Diversity: XX%
...
[GA] Generation 50/50 | Best Cost: X.XXX | P_FBR: XX.X% | Diversity: XX%

=== Final Result ===
Optimal Sequence : EVTELTLLTFHYKLR
P_FBR            : XX.X%
GRAVY            : X.XXX
Net Charge       : X.XX
hCD47 Homology   : X.XXX
Total Time       : 327s
```

---

## 📖 Development History

본 프로젝트는 초기 아이디어 구상부터 최종 최적화까지 총 9개의 주요 개발 단계를 거쳤습니다. 각 단계의 목표, 시도한 접근법, 실패한 시도, 결과를 아래 문서에서 확인할 수 있습니다.

| # | 주요 내용 | 링크 |
|---|---|---|
| 01 | 다중 오믹스 기반 → 펩타이드 서열 중심 파이프라인으로 전환 | [작업과정_1.md](docs/history/작업과정_1.md) |
| 02 | IEDB 데이터 파이프라인 구축 및 BLOSUM62 인코딩 적용 | [작업과정_2.md](docs/history/작업과정_2.md) |
| 03 | 하이브리드 예측 모델(1D-CNN + Bi-LSTM) 설계 및 손실 함수 안정화 | [작업과정_3.md](docs/history/작업과정_3.md) |
| 04 | AI 예측 엔진 + 유전 알고리즘 역설계 루프 통합 | [작업과정_4.md](docs/history/작업과정_4.md) |
| 05 | 3D SASA 데이터 통합, Smart Router 도입, 연산 최적화 | [작업과정_5.md](docs/history/작업과정_5.md) |
| 06 | XAI 대시보드 구현 (Forward Hook 기반 CNN/LSTM 활성화 분석) | [작업과정_6.md](docs/history/작업과정_6.md) |
| 07 | Ablation Study 및 모델 경량화 (1D-CNN 단독 성능 역전 발견) | [작업과정_7.md](docs/history/작업과정_7.md) |
| 08 | 데이터 스케일업(60K), 다목적 Total_Score 기반 모델 선발 | [작업과정_8.md](docs/history/작업과정_8.md) |
| 09 | 적응형 돌연변이, Min-Max 정규화, Indel 돌연변이, 최종 최적화 | [작업과정_9.md](docs/history/작업과정_9.md) |

---

## 📝 Key Engineering Decisions

1. **BLOSUM62 vs One-Hot Encoding**: 아미노산 간 진화적 치환 유사성을 보존하기 위해 BLOSUM62 채택 → 생물학적 의미 손실 없이 수치 표현 가능
2. **Pre-padding**: LSTM 시계열 특성상 실제 서열 정보가 마지막에 위치해야 Hidden State에 선명하게 전달됨
3. **Adaptive Mutation**: 고정 돌연변이율(10%) 사용 시 10세대 내 조기 수렴 확인 → 다양성 모니터링 기반 동적 조절로 해결
4. **Batch Inference**: for-loop 개별 추론 → 50개 개체군 텐서 일괄 처리로 연산 시간 목표치(10분) 달성
5. **Smart Router**: Ablation Study 결과(짧은 서열에서 CNN 단독 > Hybrid)를 역이용하여 길이 기반 동적 분기 설계

---

## 👤 Author

**Capstone Design Project** — Biomedical Engineering × AI  
Gwangju, Korea | 2026
