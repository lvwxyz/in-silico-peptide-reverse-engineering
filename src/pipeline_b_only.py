# -*- coding: utf-8 -*-
"""
In Silico Biocompatible Peptide Reverse Engineering Pipeline
=============================================================
Pipeline B: Genetic Algorithm-based Peptide Optimizer

Description:
    IEDB 데이터로 학습된 FBR 예측 모델(Pipeline A)을 심사위원으로 삼아,
    대식세포의 이물 반응(Foreign Body Response)을 최소화하는
    펩타이드 서열을 유전 알고리즘으로 역설계합니다.

Final Result:
    Optimal Sequence : EVTELTLLTFHYKLR
    P_FBR            : <목표치 이하
    KPI              : 6/6 달성

Usage:
    python pipeline_b_only.py

Author: [이경은]
Date  : 2026.05.29
"""

"""
pipeline_b_only.py (Description_Extended_Ver)
==================
학습(Ablation Study) 단계를 건너뛰고,
저장된 벤치마킹 결과(benchmark_report.csv)와 가중치(.pth)를 읽어
통합 점수로 최적 모델을 자동 선정한 뒤,
Pipeline B(GA 역설계) → 민감도 분석까지 수행하는 스크립트.

사용 조건:
  - 메인 스크립트(시스템_설계안_0528ver.py)를 한 번 이상 완전히 실행하여
    drive_path 폴더에 아래 파일들이 생성되어 있어야 함.
      · benchmark_report.csv     ← 통합 점수 산출용
      · best_<모델명>.pth 파일들  ← 각 모델 가중치
      · peptide_dataset_5_50.csv

실행 방법:
  python pipeline_b_only.py                         # 자동 선정 + 전체 실행
  python pipeline_b_only.py --skip_sensitivity       # 민감도 분석 생략
  python pipeline_b_only.py --skip_plot              # 그래프 생략
  python pipeline_b_only.py --force_model "Bi-LSTM Only"  # 모델 강제 지정
"""

import os, json, copy, random, argparse, time, tracemalloc, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 0. 경로 설정
# ──────────────────────────────────────────────────────────
drive_path = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
elif os.name == 'posix':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────────────────
# 1. 시스템 설정 (메인 스크립트와 동일하게 유지)
# ──────────────────────────────────────────────────────────
system_config = {
    "data_params": {
        "dataset_path": f"{drive_path}/peptide_dataset_5_50.csv",
        "sample_size": 60000,
        "max_len": 50
    },
    "weights": {
        "w_1_ai": 1.0,
        "w_2_gravy": 0.5,
        "w_3_charge": 0.5,
        "w_4_cd47": 0.4
    },
    "algorithm_params": {
        "ga_generations": 30,
        "ga_pop_size": 100,
        "mutation_rate": 0.3,
        "indel_rate": 0.3,
        "ga_max_len": 15,
        "checkpoint_interval": 10,
        "n_runs_bench": 3
    },
    "design_constraints": {
        "hCD47_reference": "EVTELT"
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  디바이스: {device}")

# ──────────────────────────────────────────────────────────
# 2. 모델 아키텍처 정의 (메인 스크립트와 동일)
# ──────────────────────────────────────────────────────────
class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.fc  = nn.Linear(64, 1)
    def forward(self, x, ids, mask):
        x = F.relu(self.cnn(x.permute(0,2,1)))
        return torch.sigmoid(self.fc(F.max_pool1d(x, x.size(2)).squeeze(2)))

class LSTMOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(20, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x, ids, mask):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))

class SeqHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x, ids, mask):
        c = F.relu(self.cnn(x.permute(0,2,1)))
        out, _ = self.lstm(F.max_pool1d(c, 2).permute(0,2,1))
        return torch.sigmoid(self.fc(out[:, -1, :]))

class SeqHybridNoPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x, ids, mask):
        c = F.relu(self.cnn(x.permute(0,2,1)))
        out, _ = self.lstm(c.permute(0,2,1))
        return torch.sigmoid(self.fc(out[:, -1, :]))

class ParallelHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(20, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(64+128, 1)
    def forward(self, x, ids, mask):
        xc = F.max_pool1d(F.relu(self.cnn(x.permute(0,2,1))), 50).squeeze(2)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(torch.cat((xc, out[:, -1, :]), dim=1)))

class ParallelHybridNoPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(20, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(64+128, 1)
    def forward(self, x, ids, mask):
        xc = F.relu(self.cnn(x.permute(0,2,1))).mean(dim=2)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(torch.cat((xc, out[:, -1, :]), dim=1)))

class SeqHybridResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(84, 64, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x, ids, mask):
        xc = F.relu(self.cnn(x.permute(0,2,1)))
        combined = torch.cat((x.permute(0,2,1), xc), dim=1)
        out, _ = self.lstm(combined.permute(0,2,1))
        return torch.sigmoid(self.fc(out[:, -1, :]))

class SeqHybridAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn       = nn.Conv1d(20, 64, kernel_size=3, padding=1)
        self.lstm      = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.fc        = nn.Linear(128, 1)
    def forward(self, x, ids, mask):
        c = F.relu(self.cnn(x.permute(0,2,1)))
        out, _ = self.lstm(c.permute(0,2,1))
        attn = F.softmax(self.attention(out), dim=1)
        ctx  = torch.sum(attn * out, dim=1)
        return torch.sigmoid(self.fc(ctx))

MODEL_MAP = {
    "1D-CNN Only":               CNNOnly,
    "Bi-LSTM Only":              LSTMOnly,
    "Seq Hybrid (Pool)":         SeqHybrid,
    "Seq Hybrid (No Pool)":      SeqHybridNoPool,
    "Parallel Hybrid (Pool)":    ParallelHybrid,
    "Parallel Hybrid (No Pool)": ParallelHybridNoPool,
    "Seq Hybrid (Residual)":     SeqHybridResidual,
    "Seq Hybrid (Attention)":    SeqHybridAttention,
}

# ──────────────────────────────────────────────────────────
# 3. 최적 모델 자동 선정
#    benchmark_report.csv 를 읽어 통합 점수(Total Score)를 재산출하고
#    1위 모델의 .pth 파일을 로드하여 반환.
#    --force_model 인자가 있으면 해당 모델을 강제 사용.
# ──────────────────────────────────────────────────────────
def select_and_load_best_model(force_model: str | None = None):
    csv_path = os.path.join(drive_path, "benchmark_report.csv")

    # ── 강제 지정 모드 ──
    if force_model is not None:
        print(f"ℹ️  --force_model 지정: [{force_model}]")
        model_name = force_model
    else:
        # ── CSV 없으면 .pth만으로 폴백 ──
        if not os.path.exists(csv_path):
            print(f"⚠️  {csv_path} 없음 → .pth 파일 기반 폴백 모드")
            pth_files = [f for f in os.listdir(drive_path)
                         if f.startswith("best_") and f.endswith(".pth")]
            candidates = {}
            for f in pth_files:
                raw = f[len("best_"):-len(".pth")].replace("_", " ")
                if raw in MODEL_MAP:
                    candidates[raw] = f
            if not candidates:
                raise FileNotFoundError(
                    "❌ .pth 파일도 없습니다. 메인 스크립트를 먼저 실행해 주세요."
                )
            # MODEL_MAP 순서상 첫 번째 유효 모델 선택
            for name in MODEL_MAP:
                if name in candidates:
                    model_name = name
                    break
            print(f"   폴백 선택: [{model_name}]")

        else:
            # ── CSV 기반 통합 점수 재산출 ──
            df = pd.read_csv(csv_path)

            # 메인 스크립트와 동일한 Min-Max 정규화 + 가중치
            eps = 1e-9
            W_AUROC, W_STD, W_TIME, W_MEM = 1.0, 0.1, 0.1, 0.3

            auroc_n = (df['Mean_AUROC']  - df['Mean_AUROC'].min())  / (df['Mean_AUROC'].max()  - df['Mean_AUROC'].min()  + eps)
            std_n   = (df['Std_AUROC']   - df['Std_AUROC'].min())   / (df['Std_AUROC'].max()   - df['Std_AUROC'].min()   + eps)
            time_n  = (df['Mean_Time']   - df['Mean_Time'].min())   / (df['Mean_Time'].max()   - df['Mean_Time'].min()   + eps)
            mem_n   = (df['Peak_Mem_MB'] - df['Peak_Mem_MB'].min()) / (df['Peak_Mem_MB'].max() - df['Peak_Mem_MB'].min() + eps)

            df['Total_Score'] = W_AUROC*auroc_n - W_STD*std_n - W_TIME*time_n - W_MEM*mem_n
            df_sorted = df.sort_values('Total_Score', ascending=False).reset_index(drop=True)

            print("\n📊 [모델별 통합 점수 순위]")
            print(df_sorted[['Model','Total_Score','Mean_AUROC','Std_AUROC',
                              'Mean_Time','Peak_Mem_MB']].to_string(index=False))

            # 순위 1위 모델 선정
            best_row   = df_sorted.iloc[0]
            model_name = best_row['Model']

            print(f"\n🏆 최적 모델 자동 선정: [{model_name}]")
            print(f"   통합 점수   : {best_row['Total_Score']:.4f}")
            print(f"   AUROC       : {best_row['Mean_AUROC']:.4f}  (±{best_row['Std_AUROC']:.4f})")
            print(f"   학습 시간   : {best_row['Mean_Time']:.1f} sec")
            print(f"   Peak 메모리 : {best_row['Peak_Mem_MB']:.2f} MB")

    # ── 가중치 파일 존재 확인 ──
    pth_path = os.path.join(drive_path, f"best_{model_name.replace(' ','_')}.pth")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f"❌ 가중치 파일을 찾을 수 없습니다: {pth_path}\n"
            f"   메인 스크립트에서 [{model_name}]을 학습한 뒤 다시 실행해 주세요."
        )

    # ── 모델 인스턴스화 및 가중치 로드 ──
    if model_name not in MODEL_MAP:
        raise KeyError(
            f"❌ [{model_name}]은 MODEL_MAP에 없는 모델입니다.\n"
            f"   사용 가능한 모델: {list(MODEL_MAP.keys())}"
        )

    model = MODEL_MAP[model_name]().to(device)
    state = torch.load(pth_path, map_location=device,
                       weights_only=True if int(torch.__version__.split('.')[0]) >= 2 else False)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ [{model_name}] 가중치 로드 완료 ← {pth_path}\n")
    return model, model_name


# ──────────────────────────────────────────────────────────
# 4. BLOSUM62 임베딩 레이어
# ──────────────────────────────────────────────────────────
def get_embed_layer():
    blosum62   = substitution_matrices.load("BLOSUM62")
    aa_list    = list("ACDEFGHIKLMNPQRSTVWY")
    aa_idx_map = {aa: i+1 for i, aa in enumerate(aa_list)}
    aa_idx_map['PAD'] = 0
    mat = torch.zeros((len(aa_idx_map), len(aa_list)))
    for aa1 in aa_list:
        for aa2 in aa_list:
            score = blosum62[aa1, aa2] if (aa1, aa2) in blosum62 else blosum62[aa2, aa1]
            mat[aa_idx_map[aa1], aa_idx_map[aa2]-1] = score
    layer = nn.Embedding.from_pretrained(mat, freeze=True).to(device)
    return layer, aa_idx_map


# ──────────────────────────────────────────────────────────
# 5. Pipeline B — GA 역설계 (메인 스크립트와 동일 로직)
# ──────────────────────────────────────────────────────────
def run_pipeline_B(model, aa_idx, embed_layer, config):
    algo    = config.get("algorithm_params", {})
    pop_size            = algo.get("ga_pop_size", 100)
    generations         = algo.get("ga_generations", 30)
    base_mutation_rate  = algo.get("mutation_rate", 0.1)
    indel_rate          = algo.get("indel_rate", 0.3)
    ga_max_len          = algo.get("ga_max_len", 15)
    checkpoint_interval = algo.get("checkpoint_interval", 10)
    ref         = config.get("design_constraints", {}).get("hCD47_reference", "EVTELT")
    ga_min_len  = len(ref)

    w      = config.get("weights", {})
    w_ai   = w.get("w_1_ai",    0.5)
    w_gr   = w.get("w_2_gravy", 0.15)
    w_ch   = w.get("w_3_charge",0.15)
    w_cd   = w.get("w_4_cd47",  0.2)

    # 정규화 상수
    _P_FBR_MAX  = 1.0
    _GRAVY_MAX  = 4.5
    _CHARGE_MAX = 10.0
    _MOTIF_MAX  = float(len(ref))

    history = {'gen':[], 'best_cost':[], 'mean_cost':[],
               'p_fbr':[], 'diversity':[], 'mutation_rate':[], 'indel_count':[]}

    pop = [
        ("".join(random.choices("ACDEFGHIKLMNPQRSTVWY",
                                k=max(0, random.randint(0, ga_max_len - len(ref))))) +
         ref +
         "".join(random.choices("ACDEFGHIKLMNPQRSTVWY",
                                k=max(0, random.randint(0, ga_max_len - len(ref))))))[:ga_max_len]
        for _ in range(pop_size)
    ]

    current_mutation_rate = base_mutation_rate

    for gen in range(generations):
        scored_pop = []
        for seq in pop:
            pad_len = 15 - len(seq)
            pad = ['PAD'] * pad_len + list(seq)
            with torch.no_grad():
                idx_t  = torch.tensor([[aa_idx.get(aa, 0) for aa in pad]], dtype=torch.long).to(device)
                emb_t  = embed_layer(idx_t)
                d_ids  = torch.zeros_like(idx_t)
                d_mask = torch.tensor([[0]*pad_len + [1]*len(seq)], dtype=torch.long).to(device)
                p_fbr  = model(emb_t, d_ids, d_mask).item()

            analysis = ProteinAnalysis(seq)
            gravy  = analysis.gravy()
            charge = analysis.charge_at_pH(7.4)
            h_score = sum(1 for a,b in zip(seq, ref) if a==b) if ref in seq else 0.0

            p_n = p_fbr       / _P_FBR_MAX
            g_n = abs(gravy)  / _GRAVY_MAX
            c_n = abs(charge) / _CHARGE_MAX
            h_n = h_score     / (_MOTIF_MAX + 1e-9)

            cost = (w_ai * p_n) + (w_gr * g_n) + (w_ch * c_n) - (w_cd * h_n)
            scored_pop.append((cost, p_fbr, gravy, charge, h_score, seq))

        scored_pop.sort(key=lambda x: x[0])
        best = scored_pop[0]

        costs   = [x[0] for x in scored_pop]
        div_pct = len(set(x[5] for x in scored_pop)) / pop_size * 100

        if div_pct < 15.0:
            current_mutation_rate = min(0.50, current_mutation_rate + 0.15)
            status = "⚠️ 위기 (돌연변이율 상승↑)"
        else:
            current_mutation_rate = base_mutation_rate
            status = "✅ 정상"

        history['gen'].append(gen+1)
        history['best_cost'].append(best[0])
        history['mean_cost'].append(np.mean(costs))
        history['p_fbr'].append(best[1] * 100)
        history['diversity'].append(div_pct)
        history['mutation_rate'].append(current_mutation_rate * 100)

        print(f"  [Gen {gen+1:02d}] Cost: {best[0]:.4f} | P_FBR: {best[1]*100:.1f}% | "
              f"Div: {div_pct:.1f}% | Mut: {current_mutation_rate*100:.0f}% | {status}")

        # 다음 세대 생성
        elites    = [x[5] for x in scored_pop[:max(1, int(pop_size * 0.2))]]
        next_gen  = list(elites)
        indel_cnt = 0

        while len(next_gen) < pop_size:
            p1, p2 = random.choices(scored_pop[:max(2, int(pop_size*0.5))], k=2)
            cp = random.randint(1, max(1, min(len(p1[5]), len(p2[5])) - 1))
            child = (p1[5][:cp] + p2[5][cp:])[:ga_max_len]

            if random.random() < current_mutation_rate and len(child) > 0:
                if random.random() < indel_rate:
                    if random.random() < 0.5 and len(child) < ga_max_len:
                        i = random.randint(0, len(child))
                        child = child[:i] + random.choice("ACDEFGHIKLMNPQRSTVWY") + child[i:]
                        indel_cnt += 1
                    elif len(child) > ga_min_len:
                        i = random.randint(0, len(child)-1)
                        child = child[:i] + child[i+1:]
                        indel_cnt += 1
                    else:
                        i = random.randint(0, len(child)-1)
                        child = child[:i] + random.choice("ACDEFGHIKLMNPQRSTVWY") + child[i+1:]
                else:
                    i = random.randint(0, len(child)-1)
                    child = child[:i] + random.choice("ACDEFGHIKLMNPQRSTVWY") + child[i+1:]
            next_gen.append(child)

        history['indel_count'].append(indel_cnt)
        pop = next_gen

        # 세대별 체크포인트 저장
        if (gen+1) % checkpoint_interval == 0 or gen == generations-1:
            cp_data = {
                'generation': gen+1,
                'best_seq': best[5], 'best_cost': float(best[0]),
                'best_p_fbr': float(best[1]), 'best_gravy': float(best[2]),
                'best_charge': float(best[3]), 'indel_count': indel_cnt,
                'history_snapshot': {k: [float(v) for v in vs] for k,vs in history.items()},
                'population': [x[5] for x in scored_pop]
            }
            cp_path = os.path.join(drive_path, f"ga_checkpoint_gen{gen+1:03d}.json")
            with open(cp_path, 'w', encoding='utf-8') as f:
                json.dump(cp_data, f, ensure_ascii=False, indent=2)
            print(f"  💾 [Gen {gen+1:02d}] 체크포인트 저장 → {cp_path}  (Indels: {indel_cnt})")

    final = scored_pop[0]
    return (final[:5], final[5]), history


# ──────────────────────────────────────────────────────────
# 6. 진화 그래프 시각화 (메인 스크립트와 동일)
# ──────────────────────────────────────────────────────────
def plot_evolution_history(ga_history, model_name):
    fig, axs = plt.subplots(3, 1, figsize=(14, 15), dpi=300)
    fig.suptitle(
        f"In Silico Peptide Evolution Granular Report\n"
        f"(모델: {model_name} / 적응형 유전 알고리즘 세대별 전수 추적 명세서)",
        fontsize=16, fontweight='bold', y=0.96
    )
    gens = ga_history['gen']

    axs[0].plot(gens, ga_history['best_cost'], color='#d63031', marker='o',
                markersize=5, linewidth=2.5, label='Best Individual Cost')
    axs[0].plot(gens, ga_history['mean_cost'], color='#ff7675', linestyle='--',
                linewidth=1.5, alpha=0.7, label='Population Mean Cost')
    axs[0].set_title("1. Fitness Optimization & Convergence Curve", fontsize=13, fontweight='bold')
    axs[0].set_ylabel("Total Penalty Cost Score")
    axs[0].set_xticks(gens); axs[0].grid(True, linestyle=':', alpha=0.5)
    axs[0].legend(loc='upper right', fontsize=10)

    axs[1].plot(gens, ga_history['p_fbr'], color='#0984e3', marker='s',
                markersize=5, linewidth=2.5, label='Best Individual P_FBR')
    axs[1].axhline(y=40, color='#636e72', linestyle=':', linewidth=2,
                   label='Clinical Rejection Threshold (40%)')
    axs[1].fill_between(gens, ga_history['p_fbr'], 40,
                        where=(np.array(ga_history['p_fbr']) <= 40),
                        color='#74b9ff', alpha=0.2, label='Safe Zone')
    axs[1].set_title("2. Immune Evasion Trajectory: P_FBR", fontsize=13, fontweight='bold')
    axs[1].set_ylabel("Inflammation Probability (P_FBR, %)")
    axs[1].set_xticks(gens); axs[1].set_ylim(0, max(ga_history['p_fbr']) + 10)
    axs[1].grid(True, linestyle=':', alpha=0.5); axs[1].legend(loc='upper right', fontsize=10)

    color_div = '#27ae60'
    axs[2].set_xlabel('Generation (진화 세대)', fontsize=11)
    axs[2].set_ylabel('Population Diversity (%)', color=color_div, fontsize=11)
    line_div  = axs[2].plot(gens, ga_history['diversity'], color=color_div,
                            marker='^', markersize=6, linewidth=2.5, label='Unique Sequence Ratio')
    line_warn = axs[2].axhline(y=15, color='#e74c3c', linestyle='--', linewidth=1.5,
                                alpha=0.6, label='Premature Convergence Warning (15%)')
    axs[2].tick_params(axis='y', labelcolor=color_div); axs[2].set_ylim(0, 105)

    ax_twin = axs[2].twinx()
    ax_twin.set_ylabel('Applied Mutation Rate (%)', color='#8e44ad', fontsize=11)
    bar_mut = ax_twin.bar(gens, ga_history['mutation_rate'], color='#8e44ad',
                          alpha=0.25, width=0.6, label='Dynamic Adaptive Mutation Rate')
    ax_twin.tick_params(axis='y', labelcolor='#8e44ad'); ax_twin.set_ylim(0, 60)

    handles = line_div + [line_warn, bar_mut]
    axs[2].legend(handles, [h.get_label() for h in handles], loc='upper right', fontsize=10)
    axs[2].set_title("3. Genetic Diversity Defense via Adaptive Mutation", fontsize=13, fontweight='bold')
    axs[2].set_xticks(gens); axs[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ts = datetime.now().strftime('%y%m%d_%H%M%S')
    save_path = os.path.join(drive_path, f"evolution_report_{ts}.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장: {save_path}")
    except Exception as e:
        print(f"⚠️ 그래프 저장 실패: {e}")
    plt.show()


# ──────────────────────────────────────────────────────────
# 7. 민감도 분석 (메인 스크립트와 동일)
# ──────────────────────────────────────────────────────────
def execute_sensitivity_test(model, aa_idx, embed_layer, base_config):
    scenarios = {
        "Baseline (Balanced)":      {"w_1_ai":1.0, "w_2_gravy":0.5, "w_3_charge":0.5, "w_4_cd47":0.4},
        "Scenario A (Only Immune)": {"w_1_ai":3.0, "w_2_gravy":0.1, "w_3_charge":0.1, "w_4_cd47":0.1},
        "Scenario B (Only Physio)": {"w_1_ai":0.2, "w_2_gravy":3.0, "w_3_charge":3.0, "w_4_cd47":0.1},
        "Scenario C (Motif Obsession)": {"w_1_ai":0.2, "w_2_gravy":0.1, "w_3_charge":0.1, "w_4_cd47":3.0},
    }
    model.eval()
    rows = []
    for s_name, weights in scenarios.items():
        print(f"  ▶ [{s_name}]")
        cfg = copy.deepcopy(base_config)
        cfg["weights"] = weights
        cfg["algorithm_params"]["ga_generations"] = 15
        (metrics, best_seq), _ = run_pipeline_B(model, aa_idx, embed_layer, cfg)
        cost, p_fbr, gravy, charge, motif = metrics
        rows.append({
            "Scenario":    s_name,
            "P_FBR":       f"{p_fbr*100:.1f}%",
            "GRAVY":       round(gravy, 2),
            "Charge":      round(charge, 2),
            "Motif Score": round(motif, 2),
            "Top Sequence":best_seq,
            "Cost":        round(cost, 3),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# 8. 메인 실행부
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="학습 생략 → 최적 모델 자동 선정 → Pipeline B + 민감도 분석"
    )
    parser.add_argument("--force_model", type=str, default=None,
        help='모델 강제 지정 (예: "Seq Hybrid (Pool)"). 미지정 시 CSV 기반 자동 선정.')
    parser.add_argument("--skip_sensitivity", action="store_true",
        help="민감도 분석 생략")
    parser.add_argument("--skip_plot", action="store_true",
        help="진화 그래프 생략")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("🚀 Pipeline B Only — 학습 단계 생략 모드")
    print("="*65)

    # ── (1) 최적 모델 선정 및 가중치 로드 ──────────────────
    best_model, best_model_name = select_and_load_best_model(args.force_model)

    # ── (2) 임베딩 레이어 준비 ──────────────────────────────
    embed_layer, aa_idx = get_embed_layer()

    # ── (3) Pipeline B 실행 ─────────────────────────────────
    print(f"🧬 [Pipeline B] GA 역설계 시작 (모델: {best_model_name})")
    tracemalloc.start()
    t0 = time.time()

    (metrics, best_seq), evo_history = run_pipeline_B(
        best_model, aa_idx, embed_layer, system_config)
    cost, p_fbr, gravy, charge, h_raw = metrics

    elapsed  = time.time() - t0
    _, p_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ref_len = len(system_config["design_constraints"]["hCD47_reference"])
    h_pct   = (h_raw / ref_len * 100) if ref_len > 0 else 0

    print("\n" + "="*65)
    print(f"🖥️  [Final Dashboard] — 모델: {best_model_name}")
    print("="*65)
    print(f"✨ 최종 역설계 서열 : {best_seq}")
    print("-"*65)
    print(f"  ▶ 통합 평가 점수 (Cost) : {cost:.4f}")
    print(f"  ▶ 염증 발생 확률 (P_FBR): {p_fbr*100:.2f} %")
    print(f"  ▶ 구조 상동성 (hCD47)   : {h_pct:.1f} %")
    print(f"  ▶ 친소수성 (GRAVY)      : {gravy:.3f}")
    print(f"  ▶ 전하량 (Charge)       : {charge:.3f}")
    print("-"*65)
    print(f"  ⏱️  소요 시간: {elapsed:.2f} 초  |  💾 메모리: {p_mem/1024/1024:.2f} MB")
    print("="*65)

    # ── (4) 진화 그래프 ─────────────────────────────────────
    if not args.skip_plot:
        print("\n📈 진화 과정 그래프 출력 중...")
        plot_evolution_history(evo_history, best_model_name)

    # ── (5) 민감도 분석 ─────────────────────────────────────
    if not args.skip_sensitivity:
        print("\n🧬 [Phase 3] 민감도 분석 시작...")
        df_sens = execute_sensitivity_test(best_model, aa_idx, embed_layer, system_config)
        print("\n" + "="*85)
        print(f"🏆 [Sensitivity Analysis] — 모델: {best_model_name}")
        print("="*85)
        print(df_sens.to_string(index=False))
        print("="*85)


if __name__ == "__main__":
    main()
