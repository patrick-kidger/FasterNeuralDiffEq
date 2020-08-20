import datasets
import models
import train


def main(device, norm, rtol, atol, hidden_channels=90, hidden_hidden_channels=40, num_hidden_layers=4):
    batch_size = 1024
    name = 'speech_commands/' + '-'.join(map(str, [norm, rtol, atol, hidden_channels, hidden_hidden_channels,
                                                   num_hidden_layers])).replace('.', '-')
    save = False
    max_epochs = 0
    lr = 1.6e-3
    weight_decay = 0.01

    (times, train_dataloader, val_dataloader,
     test_dataloader, input_channels, output_channels) = datasets.speech_commands(batch_size)

    model = models.NeuralCDE(times, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                             output_channels, norm, rtol, atol)

    return train.main(name, train_dataloader, val_dataloader, test_dataloader, device, model, save, max_epochs, lr,
                      weight_decay)


def full(device, norm):
    rtols = (1e-3, 1e-4, 1e-5)
    atols = (1e-6, 1e-7, 1e-8)
    for rtol, atol in zip(rtols, atols):
        for _ in range(5):
            main(device, norm, rtol, atol)
