import torch
import torch.nn as nn


def get_activation(name):
    """Return the activation function selected for the experiment."""

    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()

    raise ValueError(f"Unsupported activation function: {name}")


class VGG6(nn.Module):
    """VGG6 model used for CIFAR-10 classification."""

    def __init__(self, activation="relu", num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # First convolution block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Reduce each feature map to one value before classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # CIFAR-10 has 10 output classes
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
