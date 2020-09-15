def count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]
