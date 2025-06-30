import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from scipy.stats import norm, pearsonr

sns.set_theme(style="whitegrid", font_scale=1.2, palette="muted")
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

symbols = ['VCB', 'VNM', 'FPT']

for symbol in symbols:
    hist = pd.read_csv(f'latex_figures/{symbol}_histogram.csv')
    scatter = pd.read_csv(f'latex_figures/{symbol}_scatter.csv')
    perf = pd.read_csv(f'latex_figures/{symbol}_performance.csv')
    multi = pd.read_csv(f'latex_figures/{symbol}_multihorizon.csv')
    try:
        with open('latex_figures/comprehensive_results.json', encoding='utf-8') as f:
            comp = json.load(f)
        comp_metrics = comp['symbols'][symbol]['horizons']
    except Exception:
        comp_metrics = None
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Kết quả mô hình dự báo ngắn hạn - {symbol}', fontsize=18, fontweight='bold')
    errors = scatter['Predicted'] - scatter['True']
    sns.histplot(errors, bins=20, kde=True, stat="density", color='skyblue', ax=axs[0,0], edgecolor='gray')
    mu, std = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    axs[0,0].plot(x, norm.pdf(x, mu, std), 'r--', label=f'Normal($\mu$={mu:.3f}, $\sigma$={std:.3f})')
    axs[0,0].set_title('Phân phối sai số dự báo')
    axs[0,0].set_xlabel('Sai số')
    axs[0,0].set_ylabel('Mật độ')
    axs[0,0].legend()
    axs[0,0].axvline(0, color='k', ls=':')
    axs[0,0].annotate(f'Mean={mu:.4f}\nStd={std:.4f}', xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=11, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
    sns.scatterplot(x='True', y='Predicted', data=scatter, ax=axs[0,1], color='crimson', s=40, alpha=0.7, edgecolor='k')
    axs[0,1].plot([-0.05, 0.05], [-0.05, 0.05], 'k--', lw=1, label='Đường chéo')
    try:
        coef = np.polyfit(scatter['True'], scatter['Predicted'], 1)
        reg_x = np.linspace(scatter['True'].min(), scatter['True'].max(), 100)
        reg_y = coef[0]*reg_x + coef[1]
        axs[0,1].plot(reg_x, reg_y, 'b-', lw=2, label='Hồi quy tuyến tính')
    except Exception:
        pass
    try:
        r, _ = pearsonr(scatter['True'], scatter['Predicted'])
        axs[0,1].annotate(f'Corr = {r:.2f}', xy=(0.05, 0.92), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
    except Exception:
        pass
    axs[0,1].set_title('So sánh giá trị dự báo và thực tế')
    axs[0,1].set_xlabel('Giá trị thực tế')
    axs[0,1].set_ylabel('Giá trị dự báo')
    axs[0,1].legend()
    window = 5
    acc = perf['Direction_Accuracy'] * 100
    mse = perf['MSE']
    acc_smooth = acc.rolling(window, min_periods=1).mean()
    mse_smooth = mse.rolling(window, min_periods=1).mean()
    ax1 = axs[1,0]
    color1 = 'tab:green'
    color2 = 'tab:red'
    ax2 = ax1.twinx()
    lns1 = ax1.plot(perf['Index'], acc_smooth, color=color1, lw=2, label='Directional Accuracy (rolling mean, %)')
    ax1.set_ylabel('Độ chính xác chiều (%)', color=color1)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor=color1)
    lns2 = ax2.plot(perf['Index'], mse_smooth, color=color2, lw=2, label='MSE (rolling mean)')
    ax2.set_ylabel('MSE', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.fill_between(perf['Index'], 0, 100, where=acc_smooth < 50, color='orange', alpha=0.08)
    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')
    ax1.set_title('Độ chính xác chiều & MSE theo thời gian (rolling mean)')
    ax1.set_xlabel('Thời gian')
    ax1.grid(True, ls=':')
    width = 0.2
    horizons = multi['Horizon'].astype(str)
    x = np.arange(len(horizons))
    axs[1,1].bar(x-width, multi['Direction_Accuracy'], width, color='dodgerblue', label='Direction (%)')
    axs[1,1].bar(x, multi['MAE'], width, color='salmon', label='MAE')
    axs[1,1].bar(x+width, np.sqrt(multi['MSE']), width, color='seagreen', label='RMSE')
    axs[1,1].set_xticks(x)
    axs[1,1].set_xticklabels(horizons)
    axs[1,1].set_title('So sánh các horizon dự báo')
    axs[1,1].set_xlabel('Horizon (ngày)')
    axs[1,1].set_ylabel('Giá trị')
    axs[1,1].legend()
    axs[1,1].grid(True, ls=':')
    for i, v in enumerate(multi['Direction_Accuracy']):
        axs[1,1].text(x[i]-width, v+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, color='navy')
    for i, v in enumerate(multi['MAE']):
        axs[1,1].text(x[i], v+0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='darkred')
    for i, v in enumerate(np.sqrt(multi['MSE'])):
        axs[1,1].text(x[i]+width, v+0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10, color='darkgreen')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(top=0.90)
    plt.savefig(f'latex_figures/{symbol}_summary_figure.png', dpi=350, bbox_inches='tight')
    plt.close()