import argparse
import itertools as it
import json
import math
import matplotlib.pyplot as plt
import os
import pathlib


_here = pathlib.Path(__file__).resolve().parent


def main(dataset, models, forward):
    string = 'forward_ts' if forward else 'backward_ts'
    loc = _here / 'results' / dataset
    results = {}
    num_results = math.inf
    for folder in os.listdir(loc):
        if all(model_spec in folder for model_spec in models):
            results[folder] = []
            for run in os.listdir(loc / folder):
                with open(loc / folder / run, 'r') as f:
                    content = json.load(f)
                results[folder].append(content['test_metrics'][string])
            num_results = min(num_results, len(results[folder]))
    results = {folder: list(it.chain(*result[:num_results])) for folder, result in results.items()}

    colours = ['tab:red', 'tab:blue']
    assert len(colours) >= len(results)
    for c, (folder, ts) in reversed(list(zip(colours, results.items()))):
        plt.hist(ts, alpha=0.8, label=folder, bins=200, color=c)
    plt.xlabel('t')
    plt.ylabel('NFE at t')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('models', nargs='*')
    parser.add_argument('--forward', action='store_true')
    args = parser.parse_args()
    main(args.dataset, args.models, args.forward)
