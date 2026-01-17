# NeuroSymbolic AVO Velocity Analysis

High-resolution velocity analysis using LLM-informed Gaussian Processes to eliminate false red zones in velocity spectra.

## Features

- LLM-Driven Kernel Design: Agent dynamically synthesizes GP kernels based on data features
- Physics-Constrained Optimization: MAP estimation with physical priors
- Efficient Processing: Sparse LLM calls + spline interpolation + kernel pooling
- Rich Visualization: Industry-standard velocity spectrum plots
- Synthetic Data Generation: Built-in test data generator

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neurosymbolic-avo.git
cd neurosymbolic-avo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

### Using Synthetic Data

```bash
python scripts/process_cmp.py --synthetic --avo-type II --output synthetic_result.png
```

### Using Real SEG-Y Data

```bash
python scripts/process_cmp.py --input data/raw/stack3d.sgy --cmp 1001 --output cmp1001.png
```

### Using LLM Agent

```bash
python scripts/process_cmp.py --synthetic --use-llm --output llm_result.png
```

## Python API

```python
from neurosymbolic_avo import (
    generate_synthetic_cmp,
    process_cmp_optimized,
    plot_velocity_spectrum,
    SeismicAgent
)

# Generate synthetic data
cmp = generate_synthetic_cmp(avo_type="II")

# Process with LLM agent
agent = SeismicAgent()
velocity_spectrum = process_cmp_optimized(cmp, agent=agent)

# Visualize
plot_velocity_spectrum(velocity_spectrum, save_path="result.png")
```

## Project Structure

```
neurosymbolic-avo/
├── src/neurosymbolic_avo/    # Source code
│   ├── core/                    # Core algorithms
│   ├── agent/                   # LLM agent
│   ├── kernel/                  # Kernel factory
│   ├── solver/                  # MAP solver
│   ├── optimization/             # Performance optimization
│   ├── io/                     # Data I/O
│   └── visualization/           # Plotting
├── tests/                       # Unit and integration tests
├── scripts/                     # Command-line tools
├── docs/                        # Documentation
└── data/                        # Data directory
```

## Configuration

Edit `.env` file to configure:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.1

N_SPARSE_SAMPLES=10
KERNEL_CACHE_ENABLED=true
N_WORKERS=4
```

## Documentation

- [Vibe Coding Document](docs/design/vibe_coding_v2.0.md)
- [Technical Supplement](docs/design/technical_supplement.md)
- [API Reference](docs/api/)

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/neurosymbolic_avo --cov-report=html

# Run specific test
pytest tests/unit/test_nmo.py
```

## License

MIT License - see [LICENSE](LICENSE)
"# llm-regular" 
