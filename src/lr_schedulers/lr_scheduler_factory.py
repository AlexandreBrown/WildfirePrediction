import torch


def create_lr_scheduler(optimizer, scheduler_config: dict):
    scheduler_name = scheduler_config["name"]
    if scheduler_name == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "poly_lr":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "exponential_lr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "cosine_annealing_lr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "cyclic_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, **scheduler_config["params"]
        )
    elif scheduler_name == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **scheduler_config["params"]
        )
    else:
        raise ValueError(f"Unknown scheduler name: '{scheduler_name}'")

    return scheduler
