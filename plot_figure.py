import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据准备 - 基于Excel文件中的实际数据
data = {
    'Case1': {
        'Proposed': [60, 61, 66, 73, 70],
        'GIN': [9, 1, 1, 0, 0],
        'GIN_MI': [100, 100, 100, 100, 100],
        'GIN_impure': [57, 36, 30, 32, 28],
        'ReLvLiNGAM': [44, 54, 50, 59, 60]
    },
    'Case2': {
        'Proposed': [56, 75, 81, 93, 96],
        'GIN': [0, 0, 0, 0, 0],
        'GIN_MI': [100, 100, 100, 100, 100],
        'GIN_impure': [0, 0, 0, 0, 0],
        'ReLvLiNGAM': [52, 56, 69, 75, 78]
    },
    'Case3': {
        'Proposed': [81, 74, 86, 85, 93],
        'GIN': [90, 95, 87, 96, 87],
        'GIN_MI': [0, 0, 0, 0, 0],
        'GIN_impure': [0, 0, 0, 0, 0],
        'ReLvLiNGAM': [69, 60, 65, 64, 53]
    },
    'Case4': {
        'Proposed': [41, 29, 49, 41, 59],
        'GIN': [34, 11, 14, 16, 15],
        'GIN_MI': [0, 0, 0, 0, 0],
        'GIN_impure': [0, 0, 0, 0, 0],
        'ReLvLiNGAM': [24, 27, 33, 28, 38]
    },
    'Case5': {
        'Proposed': [32, 48, 56, 63, 84],
        'GIN': [84, 94, 89, 88, 93],
        'GIN_MI': [0, 0, 0, 0, 0],
        'GIN_impure': [0, 0, 0, 0, 0],
        'ReLvLiNGAM': [48, 34, 46, 42, 52]
    },
    'Case6': {
        'Proposed': [37, 36, 59, 76, 81],
        'GIN': [90, 94, 92, 93, 92],
        'GIN_MI': [0, 0, 0, 0, 0],
        'GIN_impure': [0, 0, 0, 0, 0],
        'ReLvLiNGAM': [26, 21, 24, 12, 18]
    }
}

# 设置参数
sample_sizes = [1000, 2000, 4000, 8000, 16000]
methods = ['Proposed', 'LiNGLaM', ' LiNGLaH', 'ReLvLiNGAM']
# methods = ['Proposed', 'GIN', 'GIN_MI', 'GIN_impure', 'ReLvLiNGAM']
cases = ['Case1', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6']

# 颜色设置
colors = ["blue", "red", "pink", "orange"]
method_colors = dict(zip(methods, colors))

# 创建2行3列的子图
fig, axes = plt.subplots(2, 3, figsize=(8, 8), sharex=True, sharey=True)
fig.suptitle('Cluster Correct Count ($N_q$) by Sample Size and Method', fontsize=12, fontweight='bold')

# 设置柱状图的宽度和位置
bar_width = 0.15
n_methods = len(methods)

for i, case in enumerate(cases):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # 计算每个方法的柱状图位置
    x_positions = np.arange(len(sample_sizes))
    
    for j, method in enumerate(methods):
        offset = (j - n_methods/2 + 0.5) * bar_width
        positions = x_positions + offset
        values = data[case][method]
        
        bars = ax.bar(positions, values, bar_width, 
                     label=method, color=method_colors[method], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置轴标签和标题
    ax.set_title(f'{case}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Sample Size', fontsize=10)
    ax.set_ylabel('$N_q$', fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{size//1000}K' for size in sample_sizes])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 添加图例（只在最后一个子图添加）
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
           ncol=len(methods), fontsize=10)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.1)

# 显示图形
plt.show()

# # 保存图形
# plt.savefig('cluster_correct_count_analysis.png', dpi=300, bbox_inches='tight')
# plt.savefig('cluster_correct_count_analysis.pdf', bbox_inches='tight')

# print("图表已生成并保存为 'cluster_correct_count_analysis.png' 和 'cluster_correct_count_analysis.pdf'")