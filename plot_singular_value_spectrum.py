'''

# 특정 layer만
python plot_singular_value_spectrum.py --layer 0

# 다른 layer
python plot_singular_value_spectrum.py --layer 15

# 전체 layer (기존 동작)
python plot_singular_value_spectrum.py

# 옵션 조합
python plot_singular_value_spectrum.py --layer 1 --top_k 150 --basis_dir ./checkpoints/phase1_XXXX/basis

'''
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 논문용 폰트 및 스타일 설정
sns.set_theme(style="white", context="paper", font_scale=1.4)

def plot_and_save_all_layers(top_k=300, target_layer=None, basis_dir=None):
    if basis_dir is None:
        basis_dir = "./checkpoints/phase1_20260429_202427/basis"
    
    # 그래프를 저장할 새 디렉토리 이름 설정
    output_dir = "./singular_value_plots"
    
    # 해당 디렉토리가 없으면 자동으로 생성 (exist_ok=True 덕분에 이미 있어도 에러 안 남)
    os.makedirs(output_dir, exist_ok=True)
    
    modules_info = [
        ('attn_q', 'attn'), 
        ('ffn_down', 'ffn_down'), 
        ('ffn_up', 'ffn_up')
    ]
    
    markers = ['o', 's', 'D'] 
    colors = sns.color_palette("husl", len(modules_info))
    
    num_layers = 32

    if target_layer is not None:
        layer_range = [target_layer]
    else:
        layer_range = range(num_layers)

    for layer_idx in layer_range:
        plt.figure(figsize=(8, 6))
        
        data_found = False # 해당 레이어에 데이터가 하나라도 있는지 체크
        
        for idx, (dir_name, label_name) in enumerate(modules_info):
            file_path = os.path.join(basis_dir, dir_name, f"layer_{layer_idx:02d}_svd.pt")
            
            if not os.path.exists(file_path):
                continue
                
            data = torch.load(file_path, map_location='cpu')
            
            if isinstance(data, dict):
                S = data.get('S', None) 
            else:
                S = data
                
            if S is None:
                continue
                
            if torch.is_tensor(S):
                S = S.numpy()
                
            data_found = True
            
            # 내림차순 정렬 및 정규화
            S = np.sort(S)[::-1]
            if S[0] > 0:
                S_normalized = S / S[0]
            else:
                S_normalized = S
                
            # 상위 top_k 개만 슬라이싱
            S_plot = S_normalized[:top_k]
            x_axis = np.arange(1, len(S_plot) + 1)
            
            plt.plot(x_axis, S_plot, marker=markers[idx], markersize=6, 
                     label=label_name, color=colors[idx], linewidth=2, markevery=max(1, top_k//30))

        if not data_found:
            print(f"Skipping Layer {layer_idx:02d}: No valid data found.")
            plt.close() # 빈 캔버스 닫기
            continue

        # 그래프 서식 설정
        plt.title(f"Singular Value Distribution (Layer {layer_idx})", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Index", fontsize=16)
        plt.ylabel("Normalized Singular Value ($S_i / S_0$)", fontsize=16)
        
        plt.xlim(0, top_k)
        plt.legend(loc='upper right', frameon=True, fontsize=14)
        
        # 테두리 설정: 위쪽과 오른쪽 테두리 제거
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 새 디렉토리 안에 파일 저장
        save_path = os.path.join(output_dir, f"layer_{layer_idx:02d}_svd_dist.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 🌟 매우 중요: 메모리 관리를 위해 생성한 그래프 닫기 (이 코드가 없으면 32장이 겹침)
        plt.close()
        
        print(f"Saved: {save_path}")

    n_saved = len(layer_range) if target_layer is None else 1
    print(f"\n✅ 완료! 총 {n_saved}개의 그래프가 '{output_dir}' 폴더에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=300, help="표시할 상위 singular value 개수")
    parser.add_argument("--layer", type=int, default=None, help="특정 layer index (미지정 시 전체 layer)")
    parser.add_argument("--basis_dir", type=str, default=None, help="basis 디렉토리 경로")
    args = parser.parse_args()
    plot_and_save_all_layers(top_k=args.top_k, target_layer=args.layer, basis_dir=args.basis_dir)