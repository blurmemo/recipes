def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output