import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def plot_cumulative_variance_specific_layer(layer_idx, top_k=500):
    basis_dir = "./checkpoints/phase1_20260427_220257/basis"
    output_dir = "./figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # attn_q, k, v는 동일하므로 통합 표기
    modules_info = [('attn_q', 'attn'), ('ffn_down', 'ffn_down'), ('ffn_up', 'ffn_up')]
    colors = sns.color_palette("husl", len(modules_info))
    
    plt.figure(figsize=(9, 6))
    
    for idx, (dir_name, label_name) in enumerate(modules_info):
        file_path = os.path.join(basis_dir, dir_name, f"layer_{layer_idx:02d}_svd.pt")
        if not os.path.exists(file_path): continue
            
        S = torch.load(file_path, map_location='cpu')
        if isinstance(S, dict): S = S.get('S', None)
        S = np.sort(S.numpy())[::-1]
        
        # 누적 비율 계산: (sigma_1 + ... + sigma_k) / total_sum
        cumulative_ratio = np.cumsum(S) / np.sum(S)
        
        # 그래프 그리기
        x_axis = np.arange(1, len(cumulative_ratio) + 1)
        plt.plot(x_axis[:top_k], cumulative_ratio[:top_k], 
                 label=label_name, color=colors[idx], linewidth=3)

    # 가이드라인 (90%, 95%) 추가
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.6, label='90% Threshold')
    plt.axhline(y=0.95, color='gray', linestyle=':', alpha=0.6, label='95% Threshold')

    plt.title(f"Cumulative Explained Variance (Layer {layer_idx})", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Singular Values (Rank)", fontsize=15)
    plt.ylabel("Cumulative Proportion of Variance", fontsize=15)
    plt.ylim(0, 1.05)
    plt.xlim(0, top_k)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"layer_{layer_idx:02d}_cumulative_variance.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Rank Collapse가 일어난 Layer 1과 일반적인 Layer 15를 그려보세요.
    plot_cumulative_variance_specific_layer(layer_idx=15, top_k=1000)