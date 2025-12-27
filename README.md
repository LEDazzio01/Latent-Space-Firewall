
# Latent Space Firewall

Latent Space Firewall is a research project exploring activation-based interventions in language models. It features a Streamlit dashboard that simulates a firewall for AI models, detecting adversarial prompts in real time using both activation analysis and keyword-based heuristics.

## Features
- **Streamlit Dashboard**: Interactive web UI for probing prompts and visualizing detection results.
- **Adversarial Prompt Detection**: Blocks prompts with adversarial intent using keyword-based logic (e.g., "delete", "shutdown").
- **Activation Harvesting**: Tools and notebooks for collecting and analyzing model activations.
- **Modular Engine**: Easily extend the firewall logic for more advanced detection or interventions.

## Project Structure

```
latent-space-firewall/
├── app.py                  # Streamlit dashboard entry point
├── data/
│   ├── raw/               # Initial JSON prompts
│   └── processed/         # Saved activations/tensors
├── notebooks/
│   └── 01_activation_harvesting.ipynb  # Exploratory analysis
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py          # Global constants (layers, thresholds)
│   ├── data_loader.py     # Prompt generation and dataset builder
│   └── firewall_engine.py # Core firewall logic
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LEDazzio01/Latent-Space-Firewall.git
   cd Latent-Space-Firewall/latent-space-firewall
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit Firewall App

```bash
streamlit run app.py
```

### Generate Sample Prompts

```bash
python -m src.data_loader
```

### Activation Analysis Notebook

Open `notebooks/01_activation_harvesting.ipynb` in Jupyter for exploratory analysis and visualization.

## How It Works

- The firewall engine uses a simple keyword-based approach to flag adversarial prompts (e.g., containing "delete", "shutdown", "bypass").
- The Streamlit UI lets you enter prompts, see detection results, and view a simulated SIEM log.
- The code is modular for easy extension to more advanced detection methods (e.g., ML classifiers on activations).

## Phases
- **Phase 1**: Activation harvesting and data collection
- **Phase 2**: Classifier training and intervention logic
- **Phase 3**: Streamlit dashboard and live demo

## License

MIT

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
