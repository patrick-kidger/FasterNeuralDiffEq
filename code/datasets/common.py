import torch


def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
