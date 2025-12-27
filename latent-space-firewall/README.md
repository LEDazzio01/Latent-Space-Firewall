# Latent Space Firewall

A research project exploring activation-based interventions in language models.

## Project Structure

```
latent-space-firewall/
├── data/
│   ├── raw/                  # Where your initial JSON prompts will live
│   └── processed/            # Where your activations.npz (tensors) will live
├── notebooks/
│   └── 01_activation_harvesting.ipynb  # For Day 2 exploratory analysis
├── src/
│   ├── __init__.py
│   ├── config.py             # Global constants (Layer numbers, Thresholds)
│   ├── data_loader.py        # Script to generate/load prompts
│   └── firewall_engine.py    # The core intervention logic (Phase 2)
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Sample Prompts

```bash
python -m src.data_loader
```

### Running the Activation Harvesting Notebook

Open `notebooks/01_activation_harvesting.ipynb` in Jupyter to begin exploratory analysis.

## Phases

- **Phase 1**: Activation harvesting and data collection
- **Phase 2**: Classifier training and intervention logic

## License

MIT
