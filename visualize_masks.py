"""
Mask Importance Visualization

Phase 2 로그 파일에서 frozen ratio를 파싱하여 시각화

Usage:
    python visualize_masks.py --log_file ./logs/phase2_20260128_002846.log
"""

import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_log_file(log_file):
    """
    Phase 2 로그 파일에서 frozen ratio 파싱
    
    로그 형식:
        Layer 0 (attn_q):
          - Frozen: 18799.0/9437184 (0.20%)
    
    Returns:
        data: dict {(layer_idx, layer_type): frozen_ratio}
        layer_types: list of layer types
        num_layers: int
    """
    print(f"Parsing log file: {log_file}")
    
    data = {}
    layer_types_set = set()
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 패턴 매칭
    # "Layer 0 (attn_q):"
    layer_pattern = re.compile(r'Layer (\d+) \((\w+)\):')
    # "  - Frozen: 18799.0/9437184 (0.20%)"
    frozen_pattern = re.compile(r'- Frozen: [\d.]+/[\d.]+ \(([\d.]+)%\)')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Layer 라인 찾기
        layer_match = layer_pattern.search(line)
        if layer_match:
            layer_idx = int(layer_match.group(1))
            layer_type = layer_match.group(2)
            layer_types_set.add(layer_type)
            
            # 다음 줄에서 Frozen 찾기
            if i + 1 < len(lines):
                frozen_match = frozen_pattern.search(lines[i + 1])
                if frozen_match:
                    frozen_ratio = float(frozen_match.group(1))
                    data[(layer_idx, layer_type)] = frozen_ratio
        
        i += 1
    
    # Layer types 정렬
    layer_types = sorted(list(layer_types_set))
    
    # Layer 개수 확인
    if data:
        num_layers = max(k[0] for k in data.keys()) + 1
    else:
        num_layers = 0
    
    print(f"✓ Parsed {len(data)} entries")
    print(f"  - Layers: 0-{num_layers-1}")
    print(f"  - Types: {layer_types}")
    
    return data, layer_types, num_layers


def create_visualizations(data, layer_types, num_layers, output_dir):
    """
    여러 종류의 시각화 생성
    
    1. Heatmap: Layer × Type
    2. Line plot: Layer별 frozen ratio 추이
    3. Bar chart: Type별 평균/최대/최소
    4. Distribution: Frozen ratio 분포
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 정리
    matrix = np.zeros((num_layers, len(layer_types)))
    
    for layer_idx in range(num_layers):
        for type_idx, layer_type in enumerate(layer_types):
            key = (layer_idx, layer_type)
            if key in data:
                matrix[layer_idx, type_idx] = data[key]
    
    # ========================================
    # 1. Heatmap: Layer × Type
    # ========================================
    plt.figure(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=layer_types,
        yticklabels=range(num_layers),
        cbar_kws={'label': 'Frozen Ratio (%)'},
        vmin=0,
        vmax=100
    )
    
    plt.title('Mask Importance Heatmap\n(Higher = More Important for Safety)', fontsize=16, fontweight='bold')
    plt.xlabel('Layer Type', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'mask_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {heatmap_path}")
    plt.close()
    
    # ========================================
    # 2. Line Plot: Layer별 추이
    # ========================================
    plt.figure(figsize=(14, 8))
    
    for type_idx, layer_type in enumerate(layer_types):
        values = [matrix[i, type_idx] for i in range(num_layers)]
        plt.plot(range(num_layers), values, marker='o', label=layer_type, linewidth=2, markersize=4)
    
    plt.title('Frozen Ratio by Layer and Type', fontsize=16, fontweight='bold')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Frozen Ratio (%)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, num_layers - 0.5)
    plt.ylim(0, 100)
    plt.xticks(range(0, num_layers, 2))
    plt.tight_layout()
    
    lineplot_path = os.path.join(output_dir, 'mask_lineplot.png')
    plt.savefig(lineplot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {lineplot_path}")
    plt.close()
    
    # ========================================
    # 3. Bar Chart: Type별 통계
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 평균
    avg_values = [matrix[:, i].mean() for i in range(len(layer_types))]
    axes[0].bar(layer_types, avg_values, color='steelblue', alpha=0.7)
    axes[0].set_title('Average Frozen Ratio by Type', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frozen Ratio (%)', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_values):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # 최대
    max_values = [matrix[:, i].max() for i in range(len(layer_types))]
    axes[1].bar(layer_types, max_values, color='coral', alpha=0.7)
    axes[1].set_title('Maximum Frozen Ratio by Type', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frozen Ratio (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(max_values):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # 최소
    min_values = [matrix[:, i].min() for i in range(len(layer_types))]
    axes[2].bar(layer_types, min_values, color='lightgreen', alpha=0.7)
    axes[2].set_title('Minimum Frozen Ratio by Type', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Frozen Ratio (%)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(min_values):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    barchart_path = os.path.join(output_dir, 'mask_statistics.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {barchart_path}")
    plt.close()
    
    # ========================================
    # 4. Distribution: Frozen ratio 분포
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for type_idx, layer_type in enumerate(layer_types):
        values = matrix[:, type_idx]
        
        axes[type_idx].hist(values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[type_idx].axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.1f}%')
        axes[type_idx].axvline(values.max(), color='orange', linestyle='--', linewidth=2, label=f'Max: {values.max():.1f}%')
        axes[type_idx].axvline(values.min(), color='green', linestyle='--', linewidth=2, label=f'Min: {values.min():.1f}%')
        
        axes[type_idx].set_title(f'{layer_type} Distribution', fontsize=14, fontweight='bold')
        axes[type_idx].set_xlabel('Frozen Ratio (%)', fontsize=12)
        axes[type_idx].set_ylabel('Count', fontsize=12)
        axes[type_idx].legend(fontsize=10)
        axes[type_idx].grid(axis='y', alpha=0.3)
    
    # 마지막 subplot 제거
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    dist_path = os.path.join(output_dir, 'mask_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dist_path}")
    plt.close()
    
    # ========================================
    # 5. 통계 텍스트 파일
    # ========================================
    stats_path = os.path.join(output_dir, 'mask_statistics.txt')
    
    with open(stats_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Mask Importance Statistics\n")
        f.write("="*70 + "\n\n")
        
        # Overall
        all_values = matrix.flatten()
        f.write("Overall Statistics:\n")
        f.write(f"  - Mean:   {all_values.mean():.2f}%\n")
        f.write(f"  - Median: {np.median(all_values):.2f}%\n")
        f.write(f"  - Std:    {all_values.std():.2f}%\n")
        f.write(f"  - Min:    {all_values.min():.2f}%\n")
        f.write(f"  - Max:    {all_values.max():.2f}%\n")
        f.write("\n")
        
        # Type별
        f.write("Statistics by Layer Type:\n")
        f.write("-"*70 + "\n")
        
        for type_idx, layer_type in enumerate(layer_types):
            values = matrix[:, type_idx]
            f.write(f"\n{layer_type}:\n")
            f.write(f"  - Mean:   {values.mean():.2f}%\n")
            f.write(f"  - Median: {np.median(values):.2f}%\n")
            f.write(f"  - Std:    {values.std():.2f}%\n")
            f.write(f"  - Min:    {values.min():.2f}% (Layer {values.argmin()})\n")
            f.write(f"  - Max:    {values.max():.2f}% (Layer {values.argmax()})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Top 10 Most Important (Highest Frozen Ratio):\n")
        f.write("-"*70 + "\n")
        
        # Top 10
        top_10 = []
        for layer_idx in range(num_layers):
            for type_idx, layer_type in enumerate(layer_types):
                value = matrix[layer_idx, type_idx]
                top_10.append((layer_idx, layer_type, value))
        
        top_10.sort(key=lambda x: x[2], reverse=True)
        
        for rank, (layer_idx, layer_type, value) in enumerate(top_10[:10], 1):
            f.write(f"{rank:2d}. Layer {layer_idx:2d} {layer_type:10s}: {value:5.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Top 10 Least Important (Lowest Frozen Ratio):\n")
        f.write("-"*70 + "\n")
        
        for rank, (layer_idx, layer_type, value) in enumerate(top_10[-10:][::-1], 1):
            f.write(f"{rank:2d}. Layer {layer_idx:2d} {layer_type:10s}: {value:5.2f}%\n")
    
    print(f"✓ Saved: {stats_path}")
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    print(f"  1. mask_heatmap.png - Layer × Type heatmap")
    print(f"  2. mask_lineplot.png - Layer-wise trend")
    print(f"  3. mask_statistics.png - Type-wise statistics")
    print(f"  4. mask_distribution.png - Distribution histograms")
    print(f"  5. mask_statistics.txt - Detailed statistics")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize mask importance from Phase 2 log file')
    parser.add_argument(
        '--log_file',
        type=str,
        required=True,
        help='Path to Phase 2 log file (e.g., ./logs/phase2_20260128_002846.log)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: ./visualizations)'
    )
    
    args = parser.parse_args()
    
    # Log 파일 확인
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    # Output 디렉토리 설정
    if args.output_dir is None:
        args.output_dir = './visualizations'
    
    print("="*70)
    print("Mask Importance Visualization (from Log File)")
    print("="*70)
    print(f"Log file: {args.log_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # 로그 파싱
    data, layer_types, num_layers = parse_log_file(args.log_file)
    
    if not data:
        print("Error: No data found in log file")
        return
    
    print()
    
    # 시각화 생성
    create_visualizations(data, layer_types, num_layers, args.output_dir)


if __name__ == '__main__':
    main()
