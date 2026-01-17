import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional


def plot_velocity_spectrum(
    semblance: np.ndarray,
    time_axis: np.ndarray,
    velocity_axis: np.ndarray,
    title: str = "Velocity Spectrum",
    picks: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    dpi: int = 300
):
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('seismic_hot', colors, N=n_bins)

    fig, ax = plt.subplots(figsize=figsize)

    extent = [velocity_axis[0], velocity_axis[-1], time_axis[-1], time_axis[0]]
    im = ax.imshow(
        semblance,
        aspect='auto',
        cmap=cmap,
        extent=extent,
        interpolation='bilinear',
        vmin=0.0,
        vmax=1.0
    )

    if picks is not None:
        ax.plot(picks, time_axis, 'k-', linewidth=2, label='Velocity Picks')
        ax.plot(picks, time_axis, 'w--', linewidth=1)

    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(im, ax=ax, label='Semblance')
    cbar.ax.tick_params(labelsize=10)

    if picks is not None:
        ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot_comparison(
    semblance_traditional: np.ndarray,
    semblance_neurosymbolic: np.ndarray,
    time_axis: np.ndarray,
    velocity_axis: np.ndarray,
    title: str = "Velocity Spectrum Comparison",
    save_path: Optional[str] = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    extent = [velocity_axis[0], velocity_axis[-1], time_axis[-1], time_axis[0]]

    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('seismic_hot', colors, N=256)

    im1 = ax1.imshow(
        semblance_traditional,
        aspect='auto',
        cmap=cmap,
        extent=extent,
        vmin=0,
        vmax=1
    )
    ax1.set_title('Traditional AB Semblance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Velocity (m/s)')
    ax1.set_ylabel('Time (s)')
    plt.colorbar(im1, ax=ax1, label='Semblance')

    im2 = ax2.imshow(
        semblance_neurosymbolic,
        aspect='auto',
        cmap=cmap,
        extent=extent,
        vmin=0,
        vmax=1
    )
    ax2.set_title('NeuroSymbolic Semblance (Ours)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Velocity (m/s)')
    ax2.set_ylabel('Time (s)')
    plt.colorbar(im2, ax=ax2, label='Semblance')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_distribution(
    features_list: list,
    feature_names: list,
    save_path: Optional[str] = None
):
    import seaborn as sns

    n_features = len(feature_names)
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, values) in enumerate(zip(feature_names, features_list)):
        sns.histplot(values, ax=axes[i], kde=True)
        axes[i].set_title(name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')

    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
