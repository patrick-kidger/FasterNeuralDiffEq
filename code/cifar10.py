import datasets
import models
import train


def main(device, norm, rtol=1e-3, atol=1e-3, hidden_channels=32, hidden_hidden_channels=128, num_pieces=8):
    batch_size = 128
    name = 'speech_commands/' + '-'.join(map(str, [norm, rtol, atol, hidden_channels, hidden_hidden_channels,
                                                   num_pieces])).replace('.', '-')
    save = True
    max_epochs = 100
    lr = 1e-2
    weight_decay = 0.01

    img_size, num_classes, train_dataloader, val_dataloader, test_dataloader = datasets.cifar10(batch_size)

    model = models.NeuralODE_CNN(img_size, num_classes, hidden_channels, hidden_hidden_channels, num_pieces, norm,
                                 rtol, atol)

    return train.main(name, train_dataloader, val_dataloader, test_dataloader, device, model, save, max_epochs, lr,
                      weight_decay)
