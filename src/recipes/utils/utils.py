import random
import torch
import numpy as np



def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def update_attrdict(obj, attrdict: dict):
    for k, v in attrdict.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def compute_model_size(model):
    """
    config: trainer config
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def print_model_status(model) -> None:
    module_states = {}
    # Iterate over all parameters
    for name, param in model.named_parameters():
        # Extract the top-level module name (e.g., "vision_model", "language_model")
        top_module = name.split(".")[0]

        # Initialize a record for the top-level module
        if top_module not in module_states:
            module_states[top_module] = {"frozen": [], "unfrozen": []}

        # Group parameters into frozen or unfrozen
        if param.requires_grad:
            module_states[top_module]["unfrozen"].append(name)
        else:
            module_states[top_module]["frozen"].append(name)

    print("--> Model state after freezing:")
    # Analyze and print the results
    for module, states in module_states.items():
        frozen_params = states["frozen"]
        unfrozen_params = states["unfrozen"]

        if frozen_params and unfrozen_params:
            # Mixed state: both frozen and unfrozen parameters
            print(f"    {module}: Frozen and Unfrozen")
        elif frozen_params:
            # All parameters are frozen
            print(f"    {module}: Frozen")
        else:
            # All parameters are unfrozen
            print(f"    {module}: Unfrozen")
