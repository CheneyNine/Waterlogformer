import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load("pred_data_3d.npy")  # shape: (24, 56, 3)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
 

# 全局取值范围统一色条
all_values = data[:, :, 2].flatten()
vmin, vmax = np.min(all_values), np.max(all_values)

# 计算 x/y 全局边界，取正方形边长
x_all = data[:, :, 0].flatten()
y_all = data[:, :, 1].flatten()
x_min, x_max = np.min(x_all), np.max(x_all)
y_min, y_max = np.min(y_all), np.max(y_all)

# 计算正方形边界框
x_range = x_max - x_min
y_range = y_max - y_min
side = max(x_range, y_range)

# 对齐中心，统一为正方形区域
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
x_min_sq = x_center - side / 2
x_max_sq = x_center + side / 2
y_min_sq = y_center - side / 2
y_max_sq = y_center + side / 2

# 插值网格数量保持相同，确保图像为正方形
n_grid = 200
xi = np.linspace(x_min_sq, x_max_sq, n_grid)
yi = np.linspace(y_min_sq, y_max_sq, n_grid)
xi, yi = np.meshgrid(xi, yi)

# 逐帧画图
for t in range(24):
    frame = data[t]
    x = frame[:, 0]
    y = frame[:, 1]
    value = frame[:, 2]

    zi = griddata((x, y), value, (xi, yi), method='cubic')

    plt.figure(figsize=(6, 6))  # 强制正方形画布
    im = plt.imshow(
        zi,
        extent=(x_min_sq, x_max_sq, y_min_sq, y_max_sq),
        origin='lower',
        cmap='turbo',
        vmin=vmin,
        vmax=vmax,
        alpha=0.95
    )
    plt.scatter(x, y, c='black', s=10, alpha=0.3)
    plt.colorbar(im, label='Predicted Value', shrink=1.0).ax.tick_params(labelsize=20)
    plt.title(f'Time Step {t + 1}', fontsize=20)
    plt.axis('off')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.savefig(f"heatmap_square_t{t+1:02d}.png")
    # plt.close()
    plt.show()  # 显示图像