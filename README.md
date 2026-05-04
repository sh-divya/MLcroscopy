MLcroscopy
===========

**Overview**
- MLcroscopy is a config-driven codebase for training and evaluating machine learning models on microscopy and XRF spectroscopy data. In its current state, the module works on materials characterized by the ADA Carbon platform. It provides dataset utilities, model architectures (classic CNN, MLP, DINO-style, and ResNet style variants), training modules, experiment sweeps, and handy scripts for analysis and visualization. Due to data scarcity, the module (which uses a pytorch and wandb backbone), cross-validation has been applied.

**Repository Structure**
- **config/**: Configuration files for models, tasks, experiments and sweeps. See [config/](config/).
- **mlcroscopy/data/**: Dataset and embedding utilities. See [mlcroscopy/data/dataset.py](mlcroscopy/data/dataset.py).
- **mlcroscopy/models/**: Model architectures and Lightning modules. See [mlcroscopy/models/pl_module.py](mlcroscopy/models/pl_module.py).
- **mlcroscopy/utils/**: Builders, constants, and miscellaneous helpers. See [mlcroscopy/utils/builders.py](mlcroscopy/utils/builders.py).
- **scripts/**: Utility scripts for baselines, embedding generation, visualization, and evaluation.

**Quickstart**
1. Create and activate a Python environment (example):

   `python -m venv .venv`

   `source .venv/bin/activate`

2. Install dependencies (if a `requirements.txt` is provided) or install common ML packages:

   `pip install .`

   or

   `pip install -r requirements.txt`

3. Run a single experiment locally with the main runner:

   `python run.py +experiments=axe`

   Use the `config/` folder to supply model/task settings via Hydra. For example, `+experiments=axe` composes [config/experiments/axe.yaml](config/experiments/axe.yaml) with the base config.

4. Run hyperparameter sweeps or multi-run experiments with the tuner:

   `python tune.py -m sweep=sweep_dino`

The code uses Hydra-style configuration. Edit or compose configs to select models, datasets, logging, and training hyperparameters.

**Experiment Configs**
- Experiment configs live in [config/experiments/](config/experiments/). They are Hydra overrides on top of [config/config.yaml](config/config.yaml), which starts from the default `classic` model, `scratch` task, and `online` logging setup.
- Run one experiment by adding the experiment group to the base config:

   `python run.py +experiments=axe`

- Run a sweep from [config/sweep/](config/sweep/) with the tuner:

   `python tune.py -m sweep=sweep_axe`

The config files are organized around the input modality and model family:

- `low_fidelity.yaml`, `plm_l1.yaml`, `plm_l2.yaml`, and `plm_smooth.yaml` use `model: mlp` with `task: expt`. These train `ExptMLP` on the five experimental descriptors only, without microscopy or XRF tensors. The `plm_*` variants keep the same MLP architecture and compare the loss function (`l1`, `l2`, or `smoothl1`).
- `axe.yaml`, `spear.yaml`, and `scimitar.yaml` use `model: classic` with `task: scratch`. These train `ScopyClassic` from microscopy images plus XRF channels. `axe` replaces pooling downsampling with stride-2 convolutions, `spear` skips the explicit downsampling block and uses stride-2 convolutions in each layer, and `scimitar` is a smaller classic CNN using global pooling.
- `trident.yaml`, `trident_l1.yaml`, `trident_l2.yaml`, and `trident_smooth.yaml` extend the best classic CNN setup with `task: both`, adding the experimental descriptors through `expt_kwargs`. The suffix variants compare the same loss choices as the MLP loss-comparison configs.
- `pike.yaml`, `halberd.yaml`, `halberd_concat.yaml`, and `halberd_vitb.yaml` use DINOv3 image features for microscopy. `pike` uses microscopy plus XRF only, while `halberd` adds experimental descriptors through `task: stego`. `halberd_concat` concatenates CLS and mean patch features, and `halberd_vitb` switches from the small ViT-S DINO backbone to the larger ViT-B backbone.
- `mace.yaml` and `morningstar.yaml` use the ResNet-style `ScopyRes` path. Both use a pretrained DINO backbone for microscopy and a residual XRF block; `morningstar` also adds experimental descriptors.
- `rapier.yaml` is currently only a placeholder comment for a larger DINO experiment and does not define a runnable override.

**Common Config Fields**
- `defaults` selects the base model, task, and logging config to override. Most experiment files use `override /model`, `override /task`, and `override /log`.
- `model.arch._target_` is the Python class Hydra instantiates, such as `ScopyClassic`, `ScopyDino`, `ScopyRes`, or `ExptMLP`.
- `micro_kwargs`, `xrf_kwargs`, and `expt_kwargs` configure the microscopy branch, XRF branch, and experimental-descriptor branch. Setting `expt_kwargs.do: True` enables the five-value experimental input branch.
- `layers` lists hidden or channel widths for a branch. For convolutional branches, the values are channel counts; for MLP branches, they are linear-layer widths.
- `kernel` and `down` control convolution and downsampling behavior. `down.type: pool` uses pooling, `down.type: conv` uses a learned stride-2 convolution, and `down.type: skip` relies on the branch convolution stride instead.
- `merge` controls how spatial feature maps become vectors before prediction. `flatten` keeps the full spatial grid, while `global` applies adaptive global average pooling.
- `pred_head_kwargs` defines the final regression head hidden widths before the two-output prediction layer.
- `model.optim` controls training behavior, including `loss`, `folds`, `epochs`, `batch_size`, `learning-rate`, `early-stopping`, and `es_patience`.

**Scripts & Utilities**
- `scripts/` contains helper tools: dataset splitting, embedding extraction, visualization, and evaluation utilities.

**TO-DO**
- In debug mode, add silent run options, where there is no results folder / printing etc.
