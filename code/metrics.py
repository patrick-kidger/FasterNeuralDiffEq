import argparse
import json
import math
import matplotlib.pyplot as plt
import os
import pathlib
import statistics


_here = pathlib.Path(__file__).resolve().parent


def main(dataset, models, nfe):
    loc = _here / 'results' / dataset
    results = {}
    num_results = math.inf
    for folder in os.listdir(loc):
        if all(model_spec in folder for model_spec in models):
            results[folder] = []
            for run in os.listdir(loc / folder):
                with open(loc / folder / run, 'r') as f:
                    content = json.load(f)
                results[folder].append([info['val_metrics']['backward_nfe' if nfe else 'accuracy']
                                        for info in content['history']])
            num_results = min(num_results, len(results[folder]))
    results = {folder: result[:num_results] for folder, result in results.items()}

    colours = ['tab:blue', 'tab:red']
    assert len(colours) >= len(results)
    max_length = max(len(l) for ll in results.values() for l in ll)
    plt.figure(figsize=(7, 2))
    for c, (folder, losses) in zip(colours, results.items()):
        if 'True' in folder:
            folder = 'Seminorm'
        else:
            folder = 'Default norm'
        # [1:] to remove outlier
        slic = slice(None if nfe else 1, None)
        mean_losses = [statistics.mean(sorted([l[i] for l in losses if len(l) > i])[slic]) for i in range(max_length)]
        std_losses = [statistics.stdev(sorted([l[i] for l in losses if len(l) > i])[slic]) for i in range(max_length)]
        upper = [m + std for m, std in zip(mean_losses, std_losses)]
        lower = [m - std for m, std in zip(mean_losses, std_losses)]
        t = range(0, 10 * max_length, 10)
        plt.fill_between(t, lower, upper, alpha=0.5, color=c)
        plt.plot(t, mean_losses, label=folder, color=c, zorder=1)
    plt.xlabel('Epoch')
    plt.ylabel('Backward NFE' if nfe else 'Accuracy')
    if not nfe:
        plt.ylim([0., 1.])
    plt.xlim([0, 200])
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('models', nargs='*')
    parser.add_argument('--nfe', action='store_true')
    args = parser.parse_args()
    main(args.dataset, args.models, args.nfe)
