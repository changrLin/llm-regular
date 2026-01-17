import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
import segyio


def visualize_segy_data(file_path):
    print(f"Visualizing SEG-Y file: {file_path}")
    
    try:
        with segyio.open(file_path, ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = len(f.trace[0])
            sample_rate = segyio.tools.dt(f) / 1000
            dt = sample_rate / 1000
            time_axis = np.arange(n_samples) * dt
            
            print(f"Loading {n_traces} traces...")
            
            data = np.zeros((n_traces, n_samples))
            cdps = []
            
            for i in range(n_traces):
                data[i, :] = f.trace[i]
                cdps.append(f.header[i][segyio.TraceField.CDP])
            
            print(f"Data shape: {data.shape}")
            print(f"Data range: {data.min():.2f} to {data.max():.2f}")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            extent = [cdps[0], cdps[-1], time_axis[-1], time_axis[0]]
            
            axes[0, 0].imshow(
                data.T,
                aspect='auto',
                cmap='seismic',
                extent=extent,
                vmin=-500,
                vmax=500
            )
            axes[0, 0].set_xlabel('CDP Number')
            axes[0, 0].set_ylabel('Time (s)')
            axes[0, 0].set_title('Seismic Section (Full)')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].imshow(
                data.T,
                aspect='auto',
                cmap='gray',
                extent=extent,
                vmin=-500,
                vmax=500
            )
            axes[0, 1].set_xlabel('CDP Number')
            axes[0, 1].set_ylabel('Time (s)')
            axes[0, 1].set_title('Seismic Section (Gray)')
            axes[0, 1].grid(True, alpha=0.3)
            
            sample_traces = np.linspace(0, n_traces - 1, 10, dtype=int)
            for i, trace_idx in enumerate(sample_traces):
                axes[1, 0].plot(data[trace_idx, :] + i * 200, time_axis, 'k-', linewidth=0.5)
            
            axes[1, 0].set_xlabel('Amplitude (scaled)')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].set_title('Sample Traces (every 100th trace)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].invert_yaxis()
            
            axes[1, 1].hist(data.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Amplitude')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Amplitude Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('SEG-Y Data Visualization', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('segy_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n✅ Visualization saved to segy_visualization.png")
            
    except Exception as e:
        print(f"Error visualizing SEG-Y file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    file_path = "test.segy"
    visualize_segy_data(file_path)
