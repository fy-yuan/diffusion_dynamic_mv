# Solving dynamic portfolio selection problems via score-based diffusion models

This repository contains all the code for the paper “Solving Dynamic Portfolio Selection Problems via Score-Based Diffusion Models” by Ahmad Aghapour, Erhan Bayraktar and Fengyi Yuan.

---

## Prerequisites

- **Python** ≥ 3.10  
- Install the required packages:

  ```bash
  pip install -r requirements.txt
  ```

## Description

This repo is organized into two main folders:

- **`synthetic_data/`** (corresponds to Subsection 4.3.1 in the paper)  
- **`real_data/`** (corresponds to Subsection 4.3.2 in the paper)  

All experiments are implemented as Jupyter notebooks.

## Usage

### Synthetic-data experiments

1. Open and run  
   [`synthetic_data/experiment.ipynb`](./synthetic_data/experiment.ipynb)  
2. If you only care about the final tables/figures, skip the **“Present training log”** cell.  
3. To fine-tune the TD3 agent, use the **“Training”** cell.  
4. For deeper modification:  
   - RL code is in [`synthetic_data/td3.py`](./synthetic_data/td3.py)  
   - Network architectures live in [`synthetic_data/core.py`](./synthetic_data/core.py)  

### Real-data experiments

1. Open and run  
   [`real_data/test_portfolios.ipynb`](./real_data/test_portfolios.ipynb)  
   - This notebook loads pre-trained models (diffusion + TD3), the processed data and saved predictions from  
     [`real_data/savings/`](./real_data/savings/)  
   - Adjust your risk-aversion parameter in the **“Prepare baselines”** cell.  
2. To re-generate or fine-tune everything from scratch:  
   1. [`real_data/prepare_data_industrial_monthly.ipynb`](./real_data/prepare_data_industrial_monthly.ipynb) downloads and preprocesses  
      `10_Industry_Portfolios_Daily.csv` (from Ken French’s data library).  
   2. [`real_data/data_and_training.ipynb`](./real_data/data_and_training.ipynb) builds training sets and trains both models.  
      - TD3 agent: **“Train the RL agent”** cell  
      - Diffusion model: **“Train the diffusion model”** cell  
   3. TD3 implementation: [`real_data/td3.py`](./real_data/td3.py)  
   4. Diffusion-model code: [`real_data/DiffusionModel/`](./real_data/DiffusionModel/)  
      - Configuration: [`real_data/DiffusionModel/config.py`](./real_data/DiffusionModel/config.py)

## Structure

```text
.
├── README.md
├── synthetic_data/
│   ├── experiment.ipynb        # reproduce synthetic-data results
│   ├── td3.py                  # TD3 implementation
│   └── core.py                 # neural-network architectures
└── real_data/
    ├── prepare_data_industrial_monthly.ipynb   # raw-data download & preprocessing
    ├── data_and_training.ipynb # data prep & model training
    ├── test_portfolios.ipynb   # load models & reproduce tables/figures
    ├── savings/                # pre-computed models, predictions, logs
    └── DiffusionModel/
        ├── config.py           # diffusion hyperparameters
        └── [other scripts]     # sampling & training routines
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Diffusion-model code is revised from [ScoreGradPred](https://github.com/yantijin/ScoreGradPred).  
- TD3 implementation is revised from [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html).  
We thank the authors for making their code publicly available.
