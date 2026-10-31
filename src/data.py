import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def create_dataloaders(batch_size=128, validation_size=5000, seed=42):
    """
    Create training, validation and test DataLoaders for CIFAR-10.

    Training data uses augmentation, while validation and test data
    use only normalization.
    """

    # Data augmentation used only for training images
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Validation and test images are not randomly augmented
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    validation_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_transform
    )

    # Create a fixed train/validation split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    ).tolist()

    # Use 45,000 images for training and 5,000 for validation
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    train_dataset = Subset(train_dataset, training_indices)
    validation_dataset = Subset(
        validation_dataset,
        validation_indices
    )

    # Create DataLoaders for training, validation and testing
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, validation_loader, test_loader
