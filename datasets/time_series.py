import collections as co
import os
import pathlib
import sklearn.model_selection
import torch
import torchcde

from . import common


_here = pathlib.Path(__file__).resolve().parent
_base_raw_data_folder = _here / 'data'
_base_processed_data_folder = _here / 'processed_data'


def _get_data_folder(name, base_data_folder):
    # We don't assume they're there by default because we might want to symlink them
    if not os.path.exists(base_data_folder):
        os.mkdir(base_data_folder)
    data_folder = base_data_folder / name
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return data_folder


def raw_data_folder(name):
    return _get_data_folder(name, _base_raw_data_folder)


def _processed_data_folder(name):
    return _get_data_folder(name, _base_processed_data_folder)


def _split_data(X, y, final_index, static):
    # 0.7/0.15/0.15 train/val/test split
    tensors = [X, y]
    if final_index is None:
        tensors.append(torch.empty(X.size(0)))
    else:
        tensors.append(final_index)
    if static is None:
        tensors.append(torch.empty(X.size(0)))
    else:
        tensors.append(static)

    train_valtest_tensors = sklearn.model_selection.train_test_split(*tensors,
                                                                     train_size=0.7,
                                                                     random_state=0,
                                                                     shuffle=True,
                                                                     stratify=y)

    (train_X, valtest_X, train_y, valtest_y, train_final_index, valtest_final_index,
     train_static, valtest_static) = train_valtest_tensors
    valtest_tensors = [valtest_X, valtest_y, valtest_final_index, valtest_static]

    val_test_tensors = sklearn.model_selection.train_test_split(*valtest_tensors,
                                                                train_size=0.5,
                                                                random_state=1,
                                                                shuffle=True,
                                                                stratify=valtest_y)
    val_X, test_X, val_y, test_y, val_final_index, test_final_index, val_static, test_static = val_test_tensors
    return (train_X, val_X, test_X, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, train_static, val_static, test_static)


def _normalise_data(X, train_X):
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def _append_extra(X, times, append_intensity):
    #####################################################
    # VERY IMPORTANT: ALWAYS APPEND TIME TO THE CONTROL #
    #####################################################
    augmented_X = [times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1), X]
    if append_intensity:
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype)
        augmented_X.append(intensity)
    return torch.cat(augmented_X, dim=2)


def _preprocess_data(times, X, y, final_index, static, output_channels, append_intensity):

    # Split data

    (train_X, val_X, test_X, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, train_static, val_static, test_static) = _split_data(X, y, final_index, static)

    # Normalise and preprocess data

    test_X = _normalise_data(test_X, train_X)
    val_X = _normalise_data(val_X, train_X)
    train_X = _normalise_data(train_X, train_X)

    test_X = _append_extra(test_X, times, append_intensity)
    val_X = _append_extra(val_X, times, append_intensity)
    train_X = _append_extra(train_X, times, append_intensity)

    train_coeffs = torchcde.linear_interpolation_coeffs(train_X, times)
    val_coeffs = torchcde.linear_interpolation_coeffs(val_X, times)
    test_coeffs = torchcde.linear_interpolation_coeffs(test_X, times)

    if static is not None:
        test_static = _normalise_data(test_static, train_static)
        val_static = _normalise_data(val_static, train_static)
        train_static = _normalise_data(train_static, train_static)

    # Return data

    train_tensors = co.OrderedDict()
    train_tensors['train_coeffs'] = train_coeffs
    train_tensors['train_y'] = train_y

    val_tensors = co.OrderedDict()
    val_tensors['val_coeffs'] = val_coeffs
    val_tensors['val_y'] = val_y

    test_tensors = co.OrderedDict()
    test_tensors['test_coeffs'] = test_coeffs
    test_tensors['test_y'] = test_y

    if final_index is not None:
        train_tensors['train_final_index'] = train_final_index
        val_tensors['val_final_index'] = val_final_index
        test_tensors['test_final_index'] = test_final_index

    if static is not None:
        train_tensors['train_static'] = train_static
        val_tensors['val_static'] = val_static
        test_tensors['test_static'] = test_static

    input_channels = torch.tensor(train_X.size(-1))
    output_channels = torch.tensor(output_channels)

    return times, train_tensors, val_tensors, test_tensors, input_channels, output_channels


def _load_data(dir):
    tensors = co.OrderedDict()
    filenames = sorted(filename for filename in os.listdir(dir))
    for filename in filenames:
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0].split('__')[1]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def _save_data(dir, tensors):
    f = '{{:0{}}}__'.format(len(str(len(tensors))))
    for i, (tensor_name, tensor_value) in enumerate(tensors.items()):
        torch.save(tensor_value, str(dir / (f.format(i) + tensor_name)) + '.pt')


def process_data(name, batch_size, download, process_data):
    processed_data_folder = _processed_data_folder(name)
    if len(os.listdir(processed_data_folder)):
        general_tensors = _load_data(processed_data_folder)
        times = general_tensors['times']
        input_channels = general_tensors['input_channels']
        output_channels = general_tensors['output_channels']
        train_tensors = _load_data(processed_data_folder / 'train')
        val_tensors = _load_data(processed_data_folder / 'val')
        test_tensors = _load_data(processed_data_folder / 'test')
    else:
        download()
        (times, train_tensors, val_tensors,
         test_tensors, input_channels, output_channels) = _preprocess_data(*process_data())
        _save_data(processed_data_folder, dict(times=times, input_channels=input_channels,
                                               output_channels=output_channels))
        os.mkdir(processed_data_folder / 'train')
        os.mkdir(processed_data_folder / 'val')
        os.mkdir(processed_data_folder / 'test')
        _save_data(processed_data_folder / 'train', train_tensors)
        _save_data(processed_data_folder / 'val', val_tensors)
        _save_data(processed_data_folder / 'test', test_tensors)

    train_dataset = torch.utils.data.TensorDataset(*train_tensors.values())
    val_dataset = torch.utils.data.TensorDataset(*val_tensors.values())
    test_dataset = torch.utils.data.TensorDataset(*test_tensors.values())

    train_dataloader = common.dataloader(train_dataset, batch_size)
    val_dataloader = common.dataloader(val_dataset, batch_size)
    test_dataloader = common.dataloader(test_dataset, batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, input_channels.item(), output_channels.item()
