def recursive_getattr(model, module_name):
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output

def recursive_setattr(model, module_name, module):
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)

def check_part_module_name(name, part_module_name):
    if isinstance(part_module_name, str):
        return part_module_name in name
    elif isinstance(part_module_name, list):
        for p_name in part_module_name:
            if p_name in name:
                return True


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warn user
                        print(f"Warning: {config_name} does not accept parameter: {k}")