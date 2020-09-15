import argparse
import json
import os
import pathlib
import torch


_here = pathlib.Path(__file__).resolve().parent


def main(name, metric):
    loc = _here / 'results' / name
    results = {}
    for folder in os.listdir(loc):
        folder_results = {metric: [],
                          'backward_nfe': []}
        for run in os.listdir(loc / folder):
            with open(loc / folder / run, 'r') as f:
                content = json.load(f)
            folder_results[metric].append(content['test_metrics'][metric])
            folder_results['backward_nfe'].append(sum(info['train_metrics']['backward_nfe']
                                                      for info in content['history']) / (10**5))
        results[folder] = {metric: torch.tensor(folder_results[metric], dtype=torch.float),
                           'backward_nfe': torch.tensor(folder_results['backward_nfe'], dtype=torch.float)}

    for folder, folder_results in results.items():
        print(f"{folder}: "
              f"Len: {len(folder_results[metric])} "
              f"{metric.capitalize()} mean: {folder_results[metric].mean():.8f} "
              f"{metric.capitalize()} std: {folder_results[metric].std():.8f} "
              f"BNFE mean (10^6): {folder_results['backward_nfe'].mean():.4f} "
              f"BNFE std (10^6): {folder_results['backward_nfe'].std():.4f} "
              f"\n\n")
            # BNFE is (10^6) because we divided by 10^5 earlier, and we only measured every 10 epochs.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('metric')
    args = parser.parse_args()
    main(args.name, args.metric)
