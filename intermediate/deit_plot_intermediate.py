"""
14/08/24
Based on 'deit_plot_performance.py' in deit dir.
Changes:
- change list of metric options.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def plot_metric(models, metrics):
    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)

        for mdl in models:
            filepath = os.path.join('out', mdl, 'log.txt')
            if os.path.exists(filepath):
                log_data = read_log_file(filepath)
                epochs = [entry['epoch'] for entry in log_data]
                values = [entry[metric] if (metric in entry) else np.nan for entry in log_data]
                plt.plot(epochs, values, linestyle='-', label=mdl)
            elif i == 0:
                print(f"Log file not found in directory: {mdl}")

        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend(title='Model')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    models = ['untrained_block11_lr0.01', 'untrained_block11_lr0.001', 'original_block11_lr0.001', 'ori_block11_lr0.01']

    # Choose: train_acc / train_loss / test_acc / test_loss
    metrics = ['train_loss', 'test_loss', 'train_acc', 'test_acc']

    plot_metric(models, metrics)
