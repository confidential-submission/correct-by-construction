# Correct-by-construction

## (fairness-preserving training with randomised response)

This repository provides an implementation of fairness-preserving training on structured datasets. The provided example runs on [Census Data](https://archive.ics.uci.edu/ml/datasets/Census+Income), but the framework can be applied to other datasets as well. This work is closely related to our manuscript **Correct-By-Construction: Certified Individual Fairness through Neural Network Training**, which provides theoretical and empirical insights into fairness-preserving training.

## Features

- Implements **randomised response** for fairness-preserving training.
- Provides multiple **training steps** including:
  - `erm_step`: [Empirical Risk Minimization (ERM)](https://en.wikipedia.org/wiki/Empirical_risk_minimization) (standard training).
  - `response_step`: Training with the [feature response algorithm](https://en.wikipedia.org/wiki/Randomized_response) for fairness.
  - `stochastic_step`: Alternative fairness-related step.
  - `binary_classification_step`: Standard [binary classification](https://en.wikipedia.org/wiki/Binary_classification).
  - `binary_fair_classification_step`: Checks consistency of predictions for [fairness evaluation](https://fairmlbook.org/).
- Supports **monitoring via [TensorBoard](https://www.tensorflow.org/tensorboard)**.
- **Model checkpoints** are saved during/after training (with easy configuration seen below).

## Installation

The required dependencies are:

```bash
pip install torch torchiteration numpy pandas
```

- [PyTorch](https://pytorch.org/): Deep learning framework.
- [torchiteration](https://pypi.org/project/torchiteration/): Iterative training support for PyTorch.
- [NumPy](https://numpy.org/): Scientific computing.
- [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.

## Usage

To run the training process on the [Census Income Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data):

```bash
python census-example.py
```

### Monitoring with [TensorBoard](https://www.tensorflow.org/tensorboard)

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir=results
tensorboard --logdir=runs
```

This command visualizes the training logs in an auto-generated folder.

## File Overview

- **[`census-example.py`](census-example.py)**: Main script for running the example on census data. The training steps can be toggled (e.g., `erm_step` or `response_step`). The number of training epochs is also configurable.
- **[`response.py`](response.py)**: Implements the **feature response algorithm** to modify input features for fairness-preserving training.
- **[`steps.py`](steps.py)**: Defines different training steps including standard ERM, feature response, stochastic, and binary classification.

## Configuration

Key training configurations (found in [`census-example.py`](census-example.py)):

- `training_step`: Specifies the training method (default: `erm_step`).
- `batch_size`: Defines the batch size for training (default: 32).
- `optimizer`: Uses **SGD** with configurable settings ([PyTorch optimizers](https://pytorch.org/docs/stable/optim.html)).
- `device`: Uses GPU (`cuda`) if available; otherwise, defaults to CPU.
- `scheduler`: Implements a [step learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

## Model Checkpoints

Trained models are saved in the `checkpoints/` directory at the end of each training epoch. The filename follows the format:

```
checkpoints/{run_name}_{epoch}.pt
```
Note that if `checkpoints/' directory happens to be missing, kindly create one.

## Dataset

By default, the example script loads [Census Income Data](https://archive.ics.uci.edu/ml/datasets/Census+Income), but it can be adapted to other datasets, e.g., German Credit, Law School Admission, etc.

## License

This project is licensed under the [MIT License](LICENSE).

---

For any questions or issues, feel free to open an issue in the [repository](https://github.com/confidential-submission/correct-by-construction/issues).
