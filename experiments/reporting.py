"""Text and plot reports for STA and DDPG comparison experiments."""

import numpy as np


def print_comparison(result_sta, result_ddpg, label=""):
    print("\n" + "=" * 60)
    if label:
        print(f"  对比结果: {label}")
    else:
        print("  对比结果 ")
    print("=" * 60)
    print(f"  {'指标':<20} {'STA-only':>12} {'STA+DDPG':>12} {'改善率':>10}")
    print("  " + "-" * 54)

    for name, key in [('IAE (绝对误差积分)', 'iae'),
                      ('ISE (平方误差积分)', 'ise'),
                      ('F   (综合指标)',     'f_value')]:
        v_sta = result_sta[key]
        v_ddpg = result_ddpg[key]
        if v_sta > 1e-12:
            improvement = (v_sta - v_ddpg) / v_sta * 100
            print(f"  {name:<18} {v_sta:>12.6f} {v_ddpg:>12.6f} {improvement:>+9.2f}%")
        else:
            print(f"  {name:<18} {v_sta:>12.6f} {v_ddpg:>12.6f} {'N/A':>10}")
    print("=" * 60)


def plot_comparison(result_sta, result_ddpg, train_f_history=None, label=""):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    from scipy.ndimage import uniform_filter1d

    time_sta = np.array(result_sta['time'])
    desired_sta = np.array(result_sta['desired'])
    actual_sta = np.array(result_sta['actual'])
    time_ddpg = np.array(result_ddpg['time'])
    desired_ddpg = np.array(result_ddpg['desired'])
    actual_ddpg = np.array(result_ddpg['actual'])

    error_sta = desired_sta - actual_sta
    error_ddpg = desired_ddpg - actual_ddpg
    dof = error_sta.shape[1]
    joint_names = [f'Joint {i}' for i in range(dof)]

    n_ds = 400
    idx_sta = np.linspace(0, len(time_sta) - 1, n_ds, dtype=int)
    idx_ddpg = np.linspace(0, len(time_ddpg) - 1, n_ds, dtype=int)
    smooth_win = max(1, len(time_sta) // n_ds)

    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig1.suptitle(f'各关节跟踪误差对比\n{label}', fontsize=14)
    for i in range(dof):
        ax = axes1[i // 2, i % 2]
        err_sta_s = uniform_filter1d(np.degrees(error_sta[:, i]),
                                     size=smooth_win, mode='nearest')
        err_ddpg_s = uniform_filter1d(np.degrees(error_ddpg[:, i]),
                                      size=smooth_win, mode='nearest')
        ax.plot(time_sta[idx_sta], err_sta_s[idx_sta],
                color='#E74C3C', alpha=0.8, linewidth=1.2, label='STA-only')
        ax.plot(time_ddpg[idx_ddpg], err_ddpg_s[idx_ddpg],
                color='#2E86C1', alpha=0.8, linewidth=1.2, label='STA+DDPG')
        ax.set_ylabel(f'{joint_names[i]} (deg)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
    axes1[2, 0].set_xlabel('Time (s)')
    axes1[2, 1].set_xlabel('Time (s)')
    fig1.tight_layout()

    total_err_sta = np.degrees(np.sum(np.abs(error_sta), axis=1))
    total_err_ddpg = np.degrees(np.sum(np.abs(error_ddpg), axis=1))

    def downsample_smooth(t, y, n_points=300, window=51):
        y_smooth = uniform_filter1d(y, size=window, mode='nearest')
        idx = np.linspace(0, len(t) - 1, n_points, dtype=int)
        return t[idx], y_smooth[idx]

    t_sta_ds, err_sta_ds = downsample_smooth(time_sta, total_err_sta)
    t_ddpg_ds, err_ddpg_ds = downsample_smooth(time_ddpg, total_err_ddpg)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.fill_between(time_sta, 0, total_err_sta, color='#E74C3C', alpha=0.08)
    ax2.fill_between(time_ddpg, 0, total_err_ddpg, color='#2E86C1', alpha=0.08)
    ax2.plot(t_sta_ds, err_sta_ds, '-o', color='#E74C3C',
             linewidth=1.8, markersize=2, alpha=0.9, label='STA-only')
    ax2.plot(t_ddpg_ds, err_ddpg_ds, '-s', color='#2E86C1',
             linewidth=1.8, markersize=2, alpha=0.9, label='STA+DDPG')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Σ|error| (deg)', fontsize=11)
    ax2.set_title(f'总跟踪误差对比\n{label}', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    metrics = ['IAE', 'ISE', 'F']
    keys = ['iae', 'ise', 'f_value']
    vals_sta = [result_sta[k] for k in keys]
    vals_ddpg = [result_ddpg[k] for k in keys]
    x = np.arange(len(metrics))
    width = 0.32
    bars1 = ax3.bar(x - width/2, vals_sta, width, label='STA-only',
                    color='#E74C3C', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, vals_ddpg, width, label='STA+DDPG',
                    color='#2E86C1', alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for i, k in enumerate(keys):
        v_s, v_d = result_sta[k], result_ddpg[k]
        if v_s > 1e-12:
            imp = (v_s - v_d) / v_s * 100
            color = '#27AE60' if imp > 0 else '#C0392B'
            ax3.annotate(f'{imp:+.2f}%',
                         xy=(x[i] + width/2, max(v_s, v_d)),
                         xytext=(0, 18), textcoords='offset points',
                         ha='center', fontsize=9, fontweight='bold', color=color,
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=11)
    ax3.set_ylabel('Value (越小越好)')
    ax3.set_title(f'性能指标对比\n{label}', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    fig3.tight_layout()

    if train_f_history:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(range(1, len(train_f_history)+1), train_f_history,
                 '-o', color='#8E44AD', linewidth=1.5, markersize=4)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('F value')
        ax4.set_title('DDPG 训练过程 F 值变化')
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()

    plt.show()
