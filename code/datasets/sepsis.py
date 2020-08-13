import csv
import math
import os
import torch
import urllib.request
import zipfile

from . import common


name = 'sepsis'


def _download():
    raw_data_folder = common.raw_data_folder(name)
    loc_Azip = raw_data_folder / 'training_setA.zip'
    loc_Bzip = raw_data_folder / 'training_setB.zip'

    if not os.path.exists(loc_Azip):
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                   str(loc_Azip))
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                   str(loc_Bzip))

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(raw_data_folder)
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(raw_data_folder)
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(raw_data_folder / folder):
                if os.path.exists(raw_data_folder / filename):
                    raise RuntimeError
                os.rename(raw_data_folder / folder / filename, raw_data_folder / filename)


def _process_data():
    raw_data_folder = common.raw_data_folder(name)

    X = []
    static = []
    y = []
    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.psv'):
            with open(raw_data_folder / filename) as file:
                point = []
                label = []
                reader = csv.reader(file, delimiter='|')
                reader = iter(reader)
                next(reader)  # first line is headings
                prev_iculos = 0
                for line in reader:
                    assert len(line) == 41
                    *time_values, age, gender, unit1, unit2, hospadmtime, iculos, sepsislabel = line
                    iculos = int(iculos)
                    if iculos > 72:  # keep at most the first three days
                        break
                    for iculos_ in range(prev_iculos + 1, iculos):
                        point.append([float('nan') for value in time_values])
                    prev_iculos = iculos
                    point.append([float(value) for value in time_values])
                    label.append(float(sepsislabel))
                unit1 = float(unit1)
                unit2 = float(unit2)
                unit1_obs = not math.isnan(unit1)
                unit2_obs = not math.isnan(unit2)
                if not unit1_obs:
                    unit1 = 0.
                if not unit2_obs:
                    unit2 = 0.
                hospadmtime = float(hospadmtime)
                if math.isnan(hospadmtime):
                    hospadmtime = 0.  # this only happens for one record
                static_ = [float(age), float(gender), unit1, unit2, hospadmtime, unit1_obs, unit2_obs]
                if len(point) > 2:
                    X.append(point)
                    static.append(static_)
                    y.append(label)
    final_index = []
    for point in X:
        final_index.append(len(point) - 1)
    maxlen = max(final_index) + 1
    for point in X:
        for _ in range(maxlen - len(point)):
            point.append([float('nan') for value in time_values])

    X = torch.tensor(X)  # shape (batch, length, input_channels)
    y = torch.tensor(y).unsqueeze(-1)  # shape (batch, length, output_channels=1)

    times = torch.linspace(0, X.size(1) - 1, X.size(1))

    final_index = torch.tensor(final_index)
    static = torch.tensor(static)
    output_channels = 2
    append_intensity = True

    return times, X, y, final_index, static, output_channels, append_intensity


def get_data(batch_size, interp_type):
    interp_y = True
    return common.get_data(name, batch_size, _download, _process_data, interp_type, interp_y)
