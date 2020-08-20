import copy
import json
import math
import numpy as np
import os
import pathlib
import time
import torch
import tqdm

from . import common

_here = pathlib.Path(__file__).resolve().parent


def _call_model(model, batch, device):
    batch = tuple(b.to(device) for b in batch)

    ret = model(*batch)
    # Normalise between models that also return accuracy, and those that don't
    if isinstance(ret, tuple):
        return ret
    else:
        return ret, torch.tensor(float('nan'), dtype=ret.dtype, device=ret.device)


def _evaluate_metrics(dataloader, model, device):
    model.eval()
    total_correct = 0
    total_dataset_size = 0
    total_loss = 0
    forward_nfe = 0
    backward_nfe = 0
    forward_ts = []
    backward_ts = []
    fwd_accept_ts = []
    fwd_reject_ts = []
    bwd_accept_ts = []
    bwd_reject_ts = []

    start_time = time.time()
    for batch in dataloader:
        model.reset_nfe_ts()
        loss, correct = _call_model(model, batch, device)
        forward_nfe += model.nfe
        forward_ts.extend(model.ts)
        accs, rejs = _count_accept_rejects(model.ts)
        fwd_accept_ts.extend(accs)
        fwd_reject_ts.extend(rejs)
        model.reset_nfe_ts()
        loss.backward()
        backward_nfe += model.nfe
        backward_ts.extend(model.ts)
        accs, rejs = _count_accept_rejects(model.ts, reverse_time=True)
        bwd_accept_ts.extend(accs)
        bwd_reject_ts.extend(rejs)
        model.zero_grad()
        model.reset_nfe_ts()

        batch_size = batch[0].size(0)
        total_correct += correct.item()
        total_dataset_size += batch_size
        total_loss += loss.item() * batch_size
    timespan = time.time() - start_time

    total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
    total_accuracy = total_correct / total_dataset_size
    metrics = common.AttrDict(accuracy=total_accuracy, dataset_size=total_dataset_size,
                              loss=total_loss, timestamp=start_time, timespan=timespan, forward_nfe=forward_nfe,
                              backward_nfe=backward_nfe, forward_ts=forward_ts, backward_ts=backward_ts,
                              fwd_accept_ts=fwd_accept_ts, fwd_reject_ts=fwd_reject_ts, bwd_accept_ts=bwd_accept_ts, bwd_reject_ts=bwd_reject_ts)
    return metrics


def _count_accept_rejects(ts, first_step_given=False, evals_per_step=6, reverse_time=False):
    if first_step_given:
        offset = evals_per_step - 1
    else:
        offset = evals_per_step + 1

    ineq = lambda t0, t1: t0 < t1 if reverse_time else t1 < t0

    step_times = ts[offset::evals_per_step]
    accepts = []
    rejects = []
    for t, next_t in zip(step_times[:-1], step_times[1:]):
        if ineq(t, next_t):
            rejects.append(t)
        else:
            accepts.append(t)
    accepts.append(step_times[-1])

    return accepts, rejects


def _train_loop(train_dataloader, val_dataloader, model, optimizer, max_epochs, device):
    model.train()
    best_model = model.state_dict()
    best_train_loss = math.inf
    best_train_loss_epoch = 0
    history = []

    epoch_per_metric = 10
    plateau_terminate = 100
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    start_time = time.time()
    for epoch in tqdm_range:
        for batch in train_dataloader:
            loss, _ = _call_model(model, batch, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            # Don't measure backward for speed?
            train_metrics = _evaluate_metrics(train_dataloader, model, device)
            val_metrics = _evaluate_metrics(val_dataloader, model, device)
            model.train()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                del best_model  # so that we don't have three copies of a model simultaneously
                best_model = copy.deepcopy(model.state_dict())

            if math.isnan(train_metrics.accuracy):
                tqdm_range.write(f'Epoch: {epoch}  '
                                 f'Train loss: {train_metrics.loss:.3}  '
                                 f'Train Forward NFE: {train_metrics.forward_nfe}  '
                                 f'Train Backward NFE: {train_metrics.backward_nfe}  '
                                 f'Val loss: {val_metrics.loss:.3}  '
                                 f'Val Forward NFE: {val_metrics.forward_nfe}  '
                                 f'Val Backward NFE: {val_metrics.backward_nfe}')
            else:
                tqdm_range.write(f'Epoch: {epoch}  '
                                 f'Train loss: {train_metrics.loss:.3}  '
                                 f'Train accuracy: {train_metrics.accuracy:.3}  '
                                 f'Train Forward NFE: {train_metrics.forward_nfe}  '
                                 f'Train Backward NFE: {train_metrics.backward_nfe}  '
                                 f'Val loss: {val_metrics.loss:.3}  '
                                 f'Val accuracy: {val_metrics.accuracy:.3}  '
                                 f'Val Forward NFE: {val_metrics.forward_nfe}  '
                                 f'Val Backward NFE: {val_metrics.backward_nfe}')
            scheduler.step(train_metrics.loss)
            history.append(common.AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                 ''.format(plateau_terminate))
                break
    timespan = time.time() - start_time

    model.load_state_dict(best_model)
    return history, start_time, timespan


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = _here / '../results' / name
    os.makedirs(loc, exist_ok=True)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)


def main(name, train_dataloader, val_dataloader, test_dataloader, device, model, save, max_epochs, lr, weight_decay):
    with torch.cuda.device(device):  # makes pin_memory work and reset_peak_memory_stats not segfault
        if device != 'cpu':
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
        else:
            baseline_memory = None

        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        history, start_time, timespan = _train_loop(train_dataloader, val_dataloader, model, optimizer, max_epochs,
                                                    device)

        model.eval()
        train_metrics = _evaluate_metrics(train_dataloader, model, device)
        val_metrics = _evaluate_metrics(val_dataloader, model, device)
        test_metrics = _evaluate_metrics(test_dataloader, model, device)

        if device != 'cpu':
            memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        else:
            memory_usage = None

    result = common.AttrDict(memory_usage=memory_usage,
                             baseline_memory=baseline_memory,
                             start_time=start_time,
                             timespan=timespan,
                             model=model.to('cpu'),
                             parameters=common.count_parameters(model),
                             history=history,
                             train_dataloader=train_dataloader,
                             val_dataloader=val_dataloader,
                             test_dataloader=test_dataloader,
                             train_metrics=train_metrics,
                             val_metrics=val_metrics,
                             test_metrics=test_metrics)
    if save:
        _save_results(name, result)
    return result
