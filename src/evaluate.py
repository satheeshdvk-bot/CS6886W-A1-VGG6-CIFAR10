import argparse

import torch
import torch.nn as nn

from .data import create_dataloaders
from .model import VGG6
from .utils import set_seed, get_device


def evaluate_model(model, loader, criterion, device):
    """Evaluate the model and return average loss and top-1 accuracy."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # No gradients are needed during final evaluation
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def parse_args():
    """Read the checkpoint path from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained VGG6 model on CIFAR-10"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved model checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()

    # Load the checkpoint and use the same configuration used for training
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]

    set_seed(config["seed"])

    # Only the test DataLoader is required for final evaluation
    _, _, test_loader = create_dataloaders(
        batch_size=config["batch_size"],
        seed=config["seed"]
    )

    model = VGG6(
        activation=config["activation"],
        num_classes=10
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    print("Final model configuration")
    print("-------------------------")
    print("Activation:", config["activation"])
    print("Optimizer:", config["optimizer"])
    print("Learning rate:", config["learning_rate"])
    print("Batch size:", config["batch_size"])
    print("Training epochs:", config["epochs"])
    print("Best checkpoint epoch:", checkpoint["epoch"])
    print("Seed:", config["seed"])
    print(
        "Best validation accuracy:",
        f'{checkpoint["validation_accuracy"]:.2f}%'
    )

    print("\nFinal test results")
    print("------------------")
    print("Test loss:", f"{test_loss:.4f}")
    print("Test top-1 accuracy:", f"{test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
