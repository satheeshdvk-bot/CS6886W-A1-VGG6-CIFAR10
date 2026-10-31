import torch.optim as optim


def get_optimizer(name, model, learning_rate):
    """Create the optimizer selected for the experiment."""

    name = name.lower()

    # Standard stochastic gradient descent
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    # SGD with momentum and Nesterov acceleration
    elif name == "nesterov":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True
        )

    # Adaptive optimizers used in the comparison
    elif name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    elif name == "adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate
        )

    elif name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif name == "nadam":
        return optim.NAdam(
            model.parameters(),
            lr=learning_rate
        )

    raise ValueError(f"Unsupported optimizer: {name}")
