import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

# 数据准备
data = {
    'Model': [
        'Waterlogformer', 'w/o static', # 'w/o rain', 
        'w/o gate fusion', 'w/o propagation', 'w/o future rain','with real rain'
    ],
    'MSE': [0.36290, 0.57459, # 0.37068, 
            0.57580, 0.48362, 0.34609, 0.44334],
    'MAE': [0.10432, 0.10658, # 0.09708, 
            0.10816, 0.13108, 0.093998, 0.10590],
    'NSE': [0.39017, 0.03444, # 0.37710, 
            0.03240, 0.18730, 0.41841, 0.25500],
    'KGE': [0.61041, 0.37252, # 0.40658, 
            -0.39807, 0.46075, 0.32083, 0.41426]
}


df = pd.DataFrame(data)
df = df.set_index('Model')

# 转换为长格式
df_long = df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')

# Remove MAE and KGE from df_long to exclude them from the plot
df_long = df_long[(df_long['Metric'] != 'MAE') & (df_long['Metric'] != 'KGE')]

# Exclude 'w/o rain' from df_long
df_long = df_long[df_long['Model'] != 'w/o rain']

# 设置风格
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 14
# 绘图
# 自定义颜色
custom_colors = [
    (0/255, 102/255, 204/255),   # 深蓝色，更高饱和度
    (34/255, 139/255, 34/255),   # 森林绿，更高饱和度
]
palette_dict = dict(zip(['MSE', 'NSE'], custom_colors))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', palette=palette_dict, ax=ax)

# Define hatch patterns and corresponding edge colors for each metric
hatches = ['////', '\\\\\\\\']

# Number of metrics and models
metrics = ['MSE', 'NSE']
n_models = df_long['Model'].nunique()

# Apply hatch fills and edge colors by metric ensuring alternating order
for j, metric in enumerate(metrics):
    for i_model in range(n_models):
        idx = j * n_models + i_model
        bar = ax.patches[idx]
        color = palette_dict[metric]
        bar.set_facecolor('white')
        bar.set_edgecolor(color)
        bar.set_hatch(hatches[j])

plt.title(' ', fontsize=16)
plt.xticks(rotation=15)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=22)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=22)
plt.xlabel('')  # 不显示 x 标签
plt.ylabel('')  # 不显示 y 标签
# Create custom legend handles to match hatch and edge colors
legend_handles = [
    Patch(facecolor='white', edgecolor=palette_dict['MSE'], hatch=hatches[0], label='MSE'),
    Patch(facecolor='white', edgecolor=palette_dict['NSE'], hatch=hatches[1], label='NSE')
]
ax.legend(handles=legend_handles, fontsize=22, title_fontsize=16)

# 设置图框边界线
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)
plt.tight_layout()

# 保存为 PDF
plt.savefig("all_metrics_comparison.pdf", format='pdf')

# 显示图像
plt.show()