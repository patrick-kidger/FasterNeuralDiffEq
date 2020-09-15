import argparse
import itertools as it
import json
import math
import matplotlib.pyplot as plt
import os
import pathlib


_here = pathlib.Path(__file__).resolve().parent


def main(dataset, models, forward, accepts, rejects):
    assert not (accepts and rejects)
    if forward:
        if accepts:
            string = 'forward_accept_ts'
        elif rejects:
            string = 'forward_reject_ts'
        else:
            string = 'forward_ts'
    else:
        if accepts:
            string = 'backward_accept_ts'
        elif rejects:
            string = 'backward_reject_ts'
        else:
            string = 'backward_ts'
    loc = _here / 'results' / dataset
    results = {}
    num_results = math.inf
    for folder in os.listdir(loc):
        if all(model_spec in folder for model_spec in models):
            results[folder] = []
            for run in os.listdir(loc / folder):
                with open(loc / folder / run, 'r') as f:
                    content = json.load(f)
                ts = []
                for info in content['history']:
                    ts.extend(info['train_metrics'][string])
                results[folder].append(ts)
            num_results = min(num_results, len(results[folder]))
    results = {folder: list(it.chain(*result[:num_results])) for folder, result in results.items()}

    colours = ['tab:red', 'tab:blue']
    assert len(colours) >= len(results)
    plt.figure(figsize=(7, 2))
    for c, (folder, ts) in (zip(colours, results.items())):
        if 'True' in folder:
            folder = 'Seminorm'
        else:
            folder = 'Default norm'
        plt.hist(ts, alpha=0.8, label=folder, bins=200, color=c)
    plt.xlabel('t')
    plt.ylabel('# Steps' if accepts or rejects else '# NFE')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('models', nargs='*')
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--accepts', action='store_true')
    parser.add_argument('--rejects', action='store_true')
    args = parser.parse_args()
    main(args.dataset, args.models, args.forward, args.accepts, args.rejects)
