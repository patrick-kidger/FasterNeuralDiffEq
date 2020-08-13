import argparse
import json
import os
import pathlib
import torch


_here = pathlib.Path(__file__).resolve().parent


def main(name):
    loc = _here / 'results' / name
    results = {}
    for folder in os.listdir(loc):
        folder_results = {'accuracy': [],
                          # 'forward_nfe': [],
                          'backward_nfe': [],
                          'timespan': []}
        for run in os.listdir(loc / folder):
            with open(loc / folder / run, 'r') as f:
                content = json.load(f)
            folder_results['accuracy'].append(content['test_metrics']['accuracy'])
            # folder_results['forward_nfe'].append(content['test_metrics']['forward_nfe'])
            folder_results['backward_nfe'].append(content['test_metrics']['backward_nfe'])
            folder_results['timespan'].append(content['timespan'] / content['history'][-1]['epoch'])
        results[folder] = {'accuracy': torch.tensor(folder_results['accuracy'], dtype=torch.float),
                           # 'forward_nfe': torch.tensor(folder_results['forward_nfe'], dtype=torch.float),
                           'backward_nfe': torch.tensor(folder_results['backward_nfe'], dtype=torch.float),
                           'timespan': torch.tensor(folder_results['timespan'], dtype=torch.float)}

    for folder, folder_results in results.items():
        print(f"{folder}: "
              f"Len: {len(folder_results['accuracy'])} "
              f"Acc mean: {folder_results['accuracy'].mean():.4f} "
              f"Acc std: {folder_results['accuracy'].std():.4f} "
              f"Timespan mean: {folder_results['timespan'].mean():.4f} "
              f"Timespan std: {folder_results['timespan'].std():.4f} "
              # f"FNFE mean: {folder_results['forward_nfe'].mean():.4f} "
              # f"FNFE std: {folder_results['forward_nfe'].std():.4f} "
              f"BNFE mean: {folder_results['backward_nfe'].mean():.4f} "
              f"BNFE std: {folder_results['backward_nfe'].std():.4f} "
              f"\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    main(args.name)
