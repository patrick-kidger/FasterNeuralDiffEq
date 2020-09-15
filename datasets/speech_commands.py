import os
import urllib.request
import tarfile
import torch
import torchaudio

from . import time_series


name = 'speech_commands'


def _download():
    raw_data_folder = time_series.raw_data_folder(name)
    loc = raw_data_folder / 'speech_commands.tar.gz'
    if os.path.exists(loc):
        return
    urllib.request.urlretrieve('http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', loc)
    with tarfile.open(loc, 'r') as f:
        f.extractall(raw_data_folder)


def _process_data():
    raw_data_folder = time_series.raw_data_folder(name)
    X = torch.empty(34975, 16000, 1)
    y = torch.empty(34975, dtype=torch.long)

    batch_index = 0
    y_index = 0
    for foldername in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
        loc = raw_data_folder / foldername
        for filename in os.listdir(loc):
            audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
                                           normalization=False)  # for forward compatbility if they fix it
            audio = audio / 2 ** 15  # Normalization argument doesn't seem to work so we do it manually.

            # A few samples are shorter than the full length; for simplicity we discard them.
            if len(audio) != 16000:
                continue

            X[batch_index] = audio
            y[batch_index] = y_index
            batch_index += 1
        y_index += 1
    assert batch_index == 34975, "batch_index is {}".format(batch_index)

    X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                   melkwargs=dict(n_fft=200, n_mels=64))(X.squeeze(-1)).transpose(1, 2).detach()

    times = torch.linspace(0, X.size(1) - 1, X.size(1))

    final_index = None
    static = None
    output_channels = y_index
    append_intensity = False

    return times, X, y, final_index, static, output_channels, append_intensity


def speech_commands(batch_size):
    return time_series.process_data(name, batch_size, _download, _process_data)
