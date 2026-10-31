import argparse
import os

import torch
import torch.nn as nn
import wandb

from .data import create_dataloaders
from .model import VGG6
from .optimizers import get_optimizer
from .utils import set_seed, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and weight update
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Calculate training accuracy during the same pass
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    """Evaluate the model without updating its weights."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Gradients are not required during validation
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


def train_model(model, train_loader, validation_loader, criterion,
                optimizer, epochs, checkpoint_path, run_config, device):
    """Train the model and save the checkpoint with best validation accuracy."""

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    for epoch in range(1, epochs + 1):

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = evaluate_model(
            model, validation_loader, criterion, device
        )

        # Keep metrics for analysis and plots
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Log the same metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Keep only the checkpoint with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_accuracy": best_val_accuracy,
                "config": run_config
            }, checkpoint_path)

    # Store the best value in the W&B run summary
    wandb.run.summary["best_val_accuracy"] = best_val_accuracy

    print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")

    return history


def parse_args():
    """Read experiment settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Train VGG6 on CIFAR-10"
    )

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--project",
        default="CS6886W-A1-VGG6-CIFAR10"
    )

    parser.add_argument(
        "--run-name",
        default="vgg6_experiment"
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize reproducibility and select the available device
    set_seed(args.seed)
    device = get_device()

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Prepare CIFAR-10 using the requested batch size
    train_loader, validation_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed
    )

    model = VGG6(
        activation=args.activation,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        args.optimizer,
        model,
        args.learning_rate
    )

    # Save the settings used for this experiment
    run_config = {
        "activation": args.activation,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.run_name}_best_model.pt"
    )

    # Start W&B tracking for this experiment
    wandb.init(
        project=args.project,
        name=args.run_name,
        config=run_config
    )

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        run_config=run_config,
        device=device
    )

    wandb.finish()


if __name__ == "__main__":
    main()
