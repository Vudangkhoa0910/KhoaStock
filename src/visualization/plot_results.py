import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from scipy.stats import norm, pearsonr

class ResultPlotter:
    def __init__(self, symbols):
        self.symbols = symbols
        self._setup_style()
        
    def _setup_style(self):
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams.update({
            'axes.titlesize': 15,
            'axes.labelsize': 13,
            'legend.fontsize': 11,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11
        })
        
    def _load_data(self, symbol):
        base = 'latex_figures'
        data = {
            'hist': pd.read_csv(f'{base}/{symbol}_histogram.csv'),
            'scatter': pd.read_csv(f'{base}/{symbol}_scatter.csv'),
            'perf': pd.read_csv(f'{base}/{symbol}_performance.csv'),
            'multi': pd.read_csv(f'{base}/{symbol}_multihorizon.csv')
        }
        
        try:
            with open(f'{base}/comprehensive_results.json') as f:
                data['comp'] = json.load(f)['symbols'][symbol]['horizons']
        except:
            data['comp'] = None
            
        return data
        
    def _plot_error_dist(self, ax, scatter):
        errors = scatter['Predicted'] - scatter['True']
        sns.histplot(errors, bins=20, kde=True, stat="density", 
                    color='skyblue', ax=ax, edgecolor='gray')
        
        mu, std = errors.mean(), errors.std()
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'r--', 
                label=f'N({mu:.3f}, {std:.3f})')
                
        ax.set_title('Forecast Error Distribution')
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()
        ax.axvline(0, color='k', ls=':')
        
        stats = f'Mean={mu:.4f}\nStd={std:.4f}'
        ax.annotate(stats, xy=(0.98, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
                   
    def _plot_scatter(self, ax, scatter):
        sns.scatterplot(x='True', y='Predicted', data=scatter, 
                       ax=ax, color='crimson', s=40, alpha=0.7)
                       
        lims = [scatter[['True', 'Predicted']].min().min(),
                scatter[['True', 'Predicted']].max().max()]
        ax.plot(lims, lims, 'k--', lw=1, label='Diagonal')
        
        try:
            coef = np.polyfit(scatter['True'], scatter['Predicted'], 1)
            x = np.linspace(lims[0], lims[1], 100)
            ax.plot(x, coef[0]*x + coef[1], 'b-', lw=2, label='Linear Regression')
            
            r, _ = pearsonr(scatter['True'], scatter['Predicted'])
            ax.annotate(f'R = {r:.2f}', xy=(0.05, 0.92), 
                       xycoords='axes fraction', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
        except:
            pass
            
        ax.set_title('Forecast vs Actual')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Forecast')
        ax.legend()
        
    def _plot_performance(self, ax, perf):
        window = 5
        acc = perf['Direction_Accuracy'] * 100
        mse = perf['MSE']
        
        acc_ma = acc.rolling(window, min_periods=1).mean()
        mse_ma = mse.rolling(window, min_periods=1).mean()
        
        ax2 = ax.twinx()
        
        l1 = ax.plot(perf['Index'], acc_ma, color='tab:green', lw=2,
                    label='Direction Accuracy MA (%)')
        l2 = ax2.plot(perf['Index'], mse_ma, color='tab:red', lw=2,
                     label='MSE MA')
                     
        ax.set_ylabel('Direction Accuracy (%)', color='tab:green')
        ax2.set_ylabel('MSE', color='tab:red')
        
        ax.set_ylim(0, 100)
        ax.tick_params(axis='y', labelcolor='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        ax.fill_between(perf['Index'], 0, 100, 
                       where=acc_ma < 50, color='orange', alpha=0.08)
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right')
        
        ax.set_title('Direction Accuracy & MSE over Time')
        ax.set_xlabel('Time')
        ax.grid(True, ls=':')
        
    def _plot_horizon(self, ax, multi):
        width = 0.2
        x = np.arange(len(multi['Horizon']))
        
        metrics = {
            'Direction': ('Direction_Accuracy', 'dodgerblue'),
            'MAE': ('MAE', 'salmon'),
            'RMSE': (lambda d: np.sqrt(d['MSE']), 'seagreen')
        }
        
        for i, (name, (metric, color)) in enumerate(metrics.items()):
            vals = multi[metric] if isinstance(metric, str) else metric(multi)
            pos = x + (i-1)*width
            
            ax.bar(pos, vals, width, color=color, label=name)
            for j, v in enumerate(vals):
                ax.text(pos[j], v+0.001, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(multi['Horizon'])
        ax.set_title('Forecast Horizon Comparison')
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, ls=':')
        
    def plot(self):
        for symbol in self.symbols:
            data = self._load_data(symbol)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 11))
            fig.suptitle(f'Short-term Forecast Results - {symbol}', 
                        fontsize=18, fontweight='bold')
                        
            self._plot_error_dist(axes[0,0], data['scatter'])
            self._plot_scatter(axes[0,1], data['scatter'])
            self._plot_performance(axes[1,0], data['perf'])
            self._plot_horizon(axes[1,1], data['multi'])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(top=0.90)
            
            plt.savefig(f'latex_figures/{symbol}_summary.png', 
                       dpi=350, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    symbols = ['VCB', 'VNM', 'FPT']
    plotter = ResultPlotter(symbols)
    plotter.plot()