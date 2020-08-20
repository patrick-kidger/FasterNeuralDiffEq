import datasets
import models
import train


def main(device, norm, rtol=1e-4, atol=1e-4):
    batch_size = 128
    name = 'acrobot/' + '-'.join(map(str, [norm, rtol, atol])).replace('.', '-')
    save = True
    max_epochs = 100
    lr = 1e-3
    weight_decay = 0.01

    times, train_dataloader, val_dataloader, test_dataloader = datasets.acrobot(batch_size)

    model = models.SymODE(times, norm, rtol, atol)
    return model

    return train.main(name, train_dataloader, val_dataloader, test_dataloader, device, model, save, max_epochs, lr,
                      weight_decay)
