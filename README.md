# Model-Based Reinforcement Learning for Energy Management in Smart Buildings

This repository contains implementations of model-based and model-free reinforcement learning algorithms applied to the CityLearn environment for smart building energy management. It includes the CHECA algorithm (CityLearn Challenge 2023 winner) and several model-based RL baselines.

---

## 🔧 Installation

Create the conda environment:

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yaml
```

---

## 📁 Project Structure

```
docs/                     # Reference papers for CityLearn and algorithms
src/
 ├── rewards/             # Reward functions for the CityLearn environment
 └── agents/
      ├── checa/          # CHECA algorithm implementation
      └── model_based/    # Model-based RL algorithms (MACURA, SAC, M2AC, MBPO)
```

## 📊 Experiment Tracking with Weights & Biases (W&B)

This project supports **Weights & Biases (W&B)** for experiment tracking, logging, and visualization.

**What is W&B?**
Weights & Biases is a platform that helps you track machine learning experiments by logging metrics, hyperparameters, configurations, and results. It provides a web dashboard to compare runs, monitor training progress, and ensure reproducibility.

### 🔑 Create an Account

1. Go to [https://wandb.ai](https://wandb.ai)
2. Create a free account (or log in if you already have one)
3. From your terminal, log in to W&B:

```bash
wandb login
```

You will be prompted to paste your API key, which you can find in your W&B account settings.

---

## 🚀 Running Experiments

All algorithms use the same environment setup.

### Activate Environment

Before running any experiment:

```bash
conda deactivate
conda activate macura_env_gymnasium_hpc_compatible
```

---

## 🟦 CHECA

```bash
cd src
python -m agents.checa.main
```

---

## 🟩 Model-Based RL Algorithms

Navigate to the model-based directory:

```bash
cd src/agents/model_based
```

Then run the desired algorithm:

### MACURA

```bash
python -m mbrl.examples.main --config-name=launcher_macura
```

### SAC

```bash
python -m mbrl.examples.main --config-name=launcher_sac
```

### M2AC

```bash
python -m mbrl.examples.main --config-name=launcher_m2ac
```

### MBPO

```bash
python -m mbrl.examples.main --config-name=launcher_mbpo
```

---

## 📈 Understanding Model-Based Algorithm Outputs

After running a model-based RL algorithm, outputs are automatically saved in a timestamped directory:

```
src/agents/model_based/outputs/{DATE}/{TIME}/
```

**Example path:**
```
src/agents/model_based/outputs/2026-01-11/22-14-11/
```

### Output Files

Each experiment generates the following files:

#### 1. **Energy Profile Visualization** - `energy_profile_{ALGO}.png`

A comprehensive energy management visualization with four subplots:
- **Building consumption components** (stacked area) with total demand overlay
- **Building demand vs. PV generation** and net load comparison
- **Battery (dis)charge control signal** over time
- **Battery state of charge (SoC)** over time

*Example: `energy_profile_sac.png`*

#### 2. **Temperature Profile Visualization** - `temperature_profile_{ALGO}.png`

Temperature analysis showing:
- **Indoor temperature** trajectory over time
- **Outdoor temperature** conditions
- **Comfort band** (setpoint ± tolerance range)

*Example: `temperature_profile_sac.png`*

#### 3. **KPI Comparison Chart** - `kpi_comparison.png`

Horizontal bar chart comparing Key Performance Indicators (KPIs) between:
- Rule-based controller (RBC) baseline
- Trained RL algorithm

#### 4. **RL Scores CSV** - `{ALGO}_rl_scores.csv`

Contains the CityLearn Challenge score breakdown:
- `comfort` - Thermal comfort performance
- `emissions` - Carbon emissions reduction
- `grid_control` - Grid stability metrics
- `resilience` - System resilience during blackout
- **Final weighted score** - Weighted sum of all components

*Example: `sac_rl_scores.csv`*

#### 5. **Complete Results Dictionary** - `{ALGO}_rl_results.pkl`

A pickled dictionary containing comprehensive experiment data:

```python
results = {
    'kpis': {
        # All Key Performance Indicators
    },
    'env_h': {
        'time_steps': [...],  # Simulation time steps
        'temperature': {
            'indoor_temperature': [...],
            'indoor_temperature_set_point': [...],
            'outdoor_temperature': [...],
            'comfort_band': [...]
        },
        'battery': {
            'soc': [...],           # State of charge
            'discharge': [...],     # Energy balance
            'consumption': [...]    # Electricity consumption
        },
        'dhw': {
            'soc': [...],           # Domestic hot water storage SoC
            'demand': [...],        # DHW demand
            'consumption': [...]    # DHW electricity consumption
        },
        'cooling_device': {
            'consumption': [...]    # Cooling electricity consumption
        },
        'net_electricity_consumption': [...],
        'solar_generation': [...],
        'non_shiftable_load': [...],
        'electricity_pricing': [...]
    }
}
```

*Example: `sac_rl_results.pkl`*

This file serves as a complete record of the experiment and can be loaded for post-processing, custom analysis, or comparison with other runs.

## ⚙️How to change parameters

> Follow the file path below 
```PATH: src/agents/model_based/mbrl/examples/conf```

> Here you can find all the `.yaml` configuration files.  

The **launcher file** defines train and test config
The **main files** define the connection between the various components.

-   The `overrides` folder contains **environment-specific parameters**.
    
-   The `dynamics_model` folder includes the configuration for the **probabilistic neural networks (PNNs)**.
    
-   The `algorithm` folder contains parameters for each specific algorithm (e.g., MACURA, MBPO, M2AC).
    