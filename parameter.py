import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

# Experiment parameters and corresponding performance metrics
drop_rates = [0.1, 0.2, 0.3]
dmodels = [256, 512, 1028]
learning_rates = [0.0001, 0.001, 0.01]
nheads = [2, 4, 8]

# Metrics for each of the above experiment settings
drop_rate_metrics = [
    (0.3629, 0.1043, 0.3902, 0.6104),
    (0.3643, 0.1105, 0.3878, 0.5119),
    (0.4580, 0.1256, 0.2304, 0.2084)
]

dmodel_metrics = [
    (0.3726, 0.1013, 0.3739, 0.3610),
    (0.3629, 0.1043, 0.3902, 0.6104),
    (0.3773, 0.0949, 0.3659, 0.6394)
]

lr_metrics = [
    (0.3629, 0.1043, 0.3902, 0.6104),
    (0.4141, 0.0767, 0.3041, 0.6520),
    (0.6051, 0.1021, -0.0169, -0.7251)
]

nheads_metrics = [
    (0.5771, 0.1117, 0.0302, -0.4596),
    (0.3629, 0.1043, 0.3902, 0.6104),
    (0.4186, 0.1067, 0.2965, 0.6328)
]

# Define font properties
font_prop = font_manager.FontProperties(family='Times New Roman')

# Create a 2x2 grid of subplots without x and y labels
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
from matplotlib import font_manager

# 修改这行：加上 size=28
font_prop = font_manager.FontProperties(family='Times New Roman', size=28)
# Plotting for Dropout Rate
indices_drop = list(range(len(drop_rates)))
axes[0, 0].plot(indices_drop, [x[0] for x in drop_rate_metrics], marker='o', markersize=10, color='tab:blue', label='MSE')
axes[0, 0].plot(indices_drop, [x[1] for x in drop_rate_metrics], marker='^', markersize=10, color='tab:orange', label='MAE')
axes[0, 0].plot(indices_drop, [x[2] for x in drop_rate_metrics], marker='*', markersize=10, color='tab:green', label='NSE')
axes[0, 0].plot(indices_drop, [x[3] for x in drop_rate_metrics], marker='s', markersize=10, color='tab:red', label='KGE')
axes[0, 0].set_xticks(indices_drop)
axes[0, 0].set_xticklabels([str(x) for x in drop_rates], fontproperties=font_prop)
axes[0, 0].set_title('(a) Dropout Rate', fontsize=36, family='Times New Roman', y=-0.25)
axes[0, 0].set_box_aspect(3/4)
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('')
axes[0, 0].tick_params(axis='both', labelsize=28)
axes[0, 0].yaxis.set_major_locator(MultipleLocator(0.2))
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font_prop)

# Plotting for dmodel
indices_dm = list(range(len(dmodels)))
axes[0, 1].plot(indices_dm, [x[0] for x in dmodel_metrics], marker='o', markersize=10, color='tab:blue', label='MSE')
axes[0, 1].plot(indices_dm, [x[1] for x in dmodel_metrics], marker='^', markersize=10, color='tab:orange', label='MAE')
axes[0, 1].plot(indices_dm, [x[2] for x in dmodel_metrics], marker='*', markersize=10, color='tab:green', label='NSE')
axes[0, 1].plot(indices_dm, [x[3] for x in dmodel_metrics], marker='s', markersize=10, color='tab:red', label='KGE')
axes[0, 1].set_xticks(indices_dm)
axes[0, 1].set_xticklabels([str(x) for x in dmodels], fontproperties=font_prop)
axes[0, 1].set_title('(b) Model Dimension', fontsize=36, family='Times New Roman', y=-0.25)
axes[0, 1].set_box_aspect(3/4)
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('')
axes[0, 1].tick_params(axis='both', labelsize=28)
axes[0, 1].yaxis.set_major_locator(MultipleLocator(0.2))
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font_prop)

# Plotting for learning rate
indices_lr = list(range(len(learning_rates)))
axes[1, 0].plot(indices_lr, [x[0] for x in lr_metrics], marker='o', markersize=10, color='tab:blue', label='MSE')
axes[1, 0].plot(indices_lr, [x[1] for x in lr_metrics], marker='^', markersize=10, color='tab:orange', label='MAE')
axes[1, 0].plot(indices_lr, [x[2] for x in lr_metrics], marker='*', markersize=10, color='tab:green', label='NSE')
axes[1, 0].plot(indices_lr, [x[3] for x in lr_metrics], marker='s', markersize=10, color='tab:red', label='KGE')
axes[1, 0].set_xticks(indices_lr)
axes[1, 0].set_xticklabels(['$\\mathregular{10^{-4}}$', '$\\mathregular{10^{-3}}$', '$\\mathregular{10^{-2}}$'], fontproperties=font_prop)
axes[1, 0].set_title('(c) Learning Rate', fontsize=36, family='Times New Roman', y=-0.25)
axes[1, 0].set_box_aspect(3/4)
axes[1, 0].legend(fontsize=26, prop=font_prop)
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('')
axes[1, 0].tick_params(axis='both', labelsize=28)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font_prop)

# Plotting for number of attention heads
indices_nh = list(range(len(nheads)))
axes[1, 1].plot(indices_nh, [x[0] for x in nheads_metrics], marker='o', markersize=10, color='tab:blue', label='MSE')
axes[1, 1].plot(indices_nh, [x[1] for x in nheads_metrics], marker='^', markersize=10, color='tab:orange', label='MAE')
axes[1, 1].plot(indices_nh, [x[2] for x in nheads_metrics], marker='*', markersize=10, color='tab:green', label='NSE')
axes[1, 1].plot(indices_nh, [x[3] for x in nheads_metrics], marker='s', markersize=10, color='tab:red', label='KGE')
axes[1, 1].set_xticks(indices_nh)
axes[1, 1].set_xticklabels([str(x) for x in nheads], fontproperties=font_prop)
axes[1, 1].set_title('(d) Number of Attention Heads', fontsize=36, family='Times New Roman', y=-0.25)
axes[1, 1].set_box_aspect(3/4)
# axes[1, 1].legend(fontsize=28, prop=font_prop)

axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('')
axes[1, 1].tick_params(axis='both', labelsize=28)
axes[1, 1].yaxis.set_major_locator(MultipleLocator(0.5))
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font_prop)

# Adjust layout
fig.tight_layout()

# Show the plot
# plt.show()
plt.savefig('parameter.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)