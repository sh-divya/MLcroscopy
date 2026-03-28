MLcroscopy
===========

**Overview**
- MLcroscopy is a lightweight, config-driven codebase for training and evaluating machine learning models on microscopy data. It provides dataset utilities, model architectures (classic, MLP, DINO-style, and concat variants), training modules, experiment sweeps, and handy scripts for analysis and visualization.

**Repository Structure**
- **config/**: Configuration files for models, tasks, and sweeps. See [config/](config/).
- **mlcroscopy/data/**: Dataset and embedding utilities. See [mlcroscopy/data/dataset.py](mlcroscopy/data/dataset.py).
- **mlcroscopy/models/**: Model architectures and Lightning modules. See [mlcroscopy/models/pl_module.py](mlcroscopy/models/pl_module.py).
- **mlcroscopy/utils/**: Builders, constants, and miscellaneous helpers. See [mlcroscopy/utils/builders.py](mlcroscopy/utils/builders.py).
- **scripts/**: Utility scripts for baselines, embedding generation, visualization, and evaluation.
- **checkpoints/**: Saved model checkpoints from training runs.
- **outputs/**: Experiment outputs and logs (organized by date).
- **slurm/**: Example Slurm job scripts for running sweeps on a cluster.

**Quickstart**
1. Create and activate a Python environment (example):

   `python -m venv .venv`

   `source .venv/bin/activate`

2. Install dependencies (if a `requirements.txt` is provided) or install common ML packages:

   `pip install -r requirements.txt`

   or

   `pip install torch torchvision pytorch-lightning hydra-core omegaconf numpy pandas scikit-learn`

3. Run a single experiment locally with the main runner:

   `python run.py`

   Use the `config/` folder to supply model/task settings via Hydra (for example, `--config-path config --config-name model/classic.yaml`).

4. Run hyperparameter sweeps or multi-run experiments with the tuner:

   `python tune.py -m config/sweep/sweep_dino.yaml`

   On clusters, use the Slurm scripts in [slurm/](slurm/) (for example `sbatch slurm/dino_sweep.slurm`).

**Configs & Experiments**
- Model definitions: [config/model/](config/model/)
- Task definitions: [config/task/](config/task/)
- Sweep definitions: [config/sweep/](config/sweep/)

The code uses Hydra-style configuration. Edit or compose configs to select models, datasets, logging, and training hyperparameters.

**Checkpoints & Outputs**
- Trained checkpoints are kept in `checkpoints/` and follow a naming convention with epoch and step metadata.
- Run outputs and logs are written to `outputs/<date>/` for easy inspection.

**Scripts & Utilities**
- `scripts/` contains helper tools: dataset splitting, embedding extraction, visualization, and evaluation utilities.
- Use `scripts/embeds.py` to compute and save embeddings for downstream analysis.

**Testing**
- Run unit tests with `pytest tests/`.

**Contributing**
- Fork, add a feature or fix, and submit a pull request. Keep changes minimal and test locally.

**License**
- Add a LICENSE file to indicate the desired license for this project.

Questions or changes you'd like in the README? I can expand sections, add example commands for specific configs, or include a requirements file.
