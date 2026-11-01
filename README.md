# CS6886W Assignment 1 - VGG6 on CIFAR-10

This repository contains the implementation and experiment results for **Assignment 1 - CS6886W Systems Engineering for Deep Learning**.

The project uses a VGG6 model on the CIFAR-10 dataset and compares different activation functions, optimizers, batch sizes, learning rates, and training epochs. Weights & Biases (W&B) was used to track the experiments and generate the required plots.

---

## Project Structure

```text
CS6886W-A1-VGG6-CIFAR10/
├── checkpoints/
│   └── best_model.pt
├── configs/
│   ├── best_config.yaml
│   └── sweep.yaml
├── results/
│   ├── plots/
│   │   ├── W&B Chart parallel_coordinates_plot.png
│   │   ├── W&B Chart validation_accuracy_vs_step.png
│   │   ├── W&B Chart Training Accuracy.png
│   │   ├── W&B Chart Training Loss.png
│   │   ├── W&B Chart Val Accuracy.png
│   │   ├── W&B Chart Val Loss.png
│   │   ├── final_accuracy_curve.png
│   │   └── final_loss_curve.png
│   ├── activation_results.csv
│   ├── optimizer_results.csv
│   ├── batch_size_results.csv
│   ├── learning_rate_results.csv
│   ├── epoch_results.csv
│   ├── baseline_history.csv
│   └── final_model_result.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── optimizers.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── README.md
└── requirements.txt
```

---

## Dataset and Preprocessing

The experiments use the **CIFAR-10** dataset.

The original training set is split into:

- 45,000 training images
- 5,000 validation images

The CIFAR-10 test set contains 10,000 images.

The following augmentation is applied to the training images:

- Random crop of 32x32 with padding of 4
- Random horizontal flip
- Conversion to tensor
- CIFAR-10 normalization

Validation and test images use only tensor conversion and normalization.

Normalization values used:

```text
Mean = (0.4914, 0.4822, 0.4465)
Std  = (0.2023, 0.1994, 0.2010)
```

A fixed seed is used for the train/validation split so that the same split can be reproduced.

---

## Model

The model is a VGG6-style convolutional neural network for CIFAR-10 classification.

The network contains:

```text
Conv2D 3 -> 64
BatchNorm
Activation

Conv2D 64 -> 64
BatchNorm
Activation
MaxPool

Conv2D 64 -> 128
BatchNorm
Activation

Conv2D 128 -> 128
BatchNorm
Activation
MaxPool

Adaptive Average Pooling
Fully Connected Layer 128 -> 10
```

The model supports the following activation functions:

- ReLU
- Sigmoid
- Tanh
- SiLU
- GELU

---

## Activation Function Comparison

For this experiment, the optimizer was fixed as Adam with learning rate `0.001`, batch size `128`, and 10 epochs.

| Activation | Best Validation Accuracy |
|---|---:|
| ReLU | 74.04% |
| Sigmoid | 36.40% |
| Tanh | 68.52% |
| SiLU | 74.08% |
| GELU | **75.88%** |

GELU gave the best validation accuracy among the tested activation functions.

---

## Optimizer Comparison

The optimizer experiments used GELU as the activation function with learning rate `0.001`, batch size `128`, and 10 epochs.

| Optimizer | Best Validation Accuracy |
|---|---:|
| SGD | 42.96% |
| Nesterov-SGD | 61.06% |
| Adam | 75.88% |
| Adagrad | 60.96% |
| RMSprop | 73.44% |
| NAdam | **76.64%** |

NAdam gave the best result in this comparison.

The same learning rate was used for all optimizers so that the comparison remained consistent. Individual optimizers such as SGD may perform differently with separately tuned learning rates.

---

## Batch Size, Learning Rate and Epoch Experiments

### Batch Size

Using GELU, NAdam, learning rate `0.001`, and 10 epochs:

| Batch Size | Best Validation Accuracy |
|---:|---:|
| 64 | 73.98% |
| 128 | **76.64%** |
| 256 | 70.92% |

Batch size `128` gave the best result.

### Learning Rate

Using GELU, NAdam, batch size `128`, and 10 epochs:

| Learning Rate | Best Validation Accuracy |
|---:|---:|
| 0.0001 | 66.84% |
| 0.001 | 76.64% |
| 0.01 | **79.48%** |

Learning rate `0.01` gave the best result.

### Epochs

Using GELU, NAdam, learning rate `0.01`, and batch size `128`:

| Epochs | Best Validation Accuracy |
|---:|---:|
| 10 | 79.48% |
| 20 | 85.66% |
| 40 | 86.82% |
| 60 | **87.42%** |

The 60-epoch run gave the highest validation accuracy. The best checkpoint within this run was obtained at epoch 47.

---

## Best Configuration and Final Result

The final configuration selected from the experiments was:

| Parameter | Value |
|---|---|
| Model | VGG6 |
| Dataset | CIFAR-10 |
| Activation | GELU |
| Optimizer | NAdam |
| Learning Rate | 0.01 |
| Batch Size | 128 |
| Training Epochs | 60 |
| Best Checkpoint Epoch | 47 |
| Seed | 42 |

Final results:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **87.42%** |
| Final Test Top-1 Accuracy | **85.94%** |
| Final Test Loss | **0.5217** |

Although the training run was configured for 60 epochs, the checkpoint from epoch 47 was used for final testing because it had the best validation accuracy.

The later epochs showed increasing training accuracy while validation accuracy started to plateau and fluctuate, indicating some overfitting after the best checkpoint.

The same configuration is stored in:

```text
configs/best_config.yaml
```

---

## Experiment Tracking

Weights & Biases was used to record:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Experiment configuration
- Best validation accuracy

W&B project:

https://wandb.ai/cs25m534-iitm/CS6886W-A1-VGG6-CIFAR10

The exported assignment plots are available under:

```text
results/plots/
```

These include:

- Parallel coordinates plot
- Validation accuracy vs step
- Training accuracy
- Training loss
- Validation accuracy
- Validation loss
- Final accuracy curve
- Final loss curve

---

## Environment

The experiments were implemented using the following environment:

```text
Python             3.13.18
PyTorch            2.11.0
Torchvision        0.26.0
NumPy              2.1.3
Pandas             2.2.3
Matplotlib         3.10.0
Weights & Biases   0.28.1
```

The experiments were run in Google Colab. GPU acceleration was used during model training.

The code automatically uses CUDA when a GPU is available and falls back to CPU otherwise.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/satheeshdvk-bot/CS6886W-A1-VGG6-CIFAR10.git
cd CS6886W-A1-VGG6-CIFAR10
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

The training script accepts the experiment settings through command-line arguments.

To reproduce the final configuration:

```bash
python -m src.train \
  --activation gelu \
  --optimizer nadam \
  --learning-rate 0.01 \
  --batch-size 128 \
  --epochs 60 \
  --seed 42 \
  --run-name final_gelu_nadam
```

During training, the script:

- prepares the CIFAR-10 train and validation data
- creates the VGG6 model
- creates the selected optimizer
- trains the model
- evaluates validation performance after each epoch
- logs metrics to W&B
- saves the checkpoint with the highest validation accuracy

---

## Evaluation

The best checkpoint included in this repository can be evaluated using:

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pt
```

Expected output is approximately:

```text
Final model configuration
-------------------------
Activation: gelu
Optimizer: nadam
Learning rate: 0.01
Batch size: 128
Training epochs: 60
Best checkpoint epoch: 47
Seed: 42
Best validation accuracy: 87.42%

Final test results
------------------
Test loss: 0.5217
Test top-1 accuracy: 85.94%
```

The evaluation script reads the saved configuration from the checkpoint, recreates the model, loads the trained weights, and evaluates the model on the CIFAR-10 test set.

---

## Reproducibility

A fixed seed of `42` is used for:

- Python
- NumPy
- PyTorch
- CUDA random number generation
- Train/validation split

The repository also includes:

```text
requirements.txt
configs/best_config.yaml
configs/sweep.yaml
checkpoints/best_model.pt
```

These files provide the dependency versions, final experiment configuration, and trained model required to reproduce the result.
configs/sweep.yaml contains the W&B sweep search space used to reproduce the hyperparameter exploration

---

## Result Files

The experiment results are stored under `results/`.

| File | Description |
|---|---|
| `activation_results.csv` | Activation function comparison |
| `optimizer_results.csv` | Optimizer comparison |
| `batch_size_results.csv` | Batch size comparison |
| `learning_rate_results.csv` | Learning rate comparison |
| `epoch_results.csv` | Epoch comparison |
| `baseline_history.csv` | Baseline training history |
| `final_model_result.csv` | Final test loss and accuracy |

The trained model is available at:

```text
checkpoints/best_model.pt
```
