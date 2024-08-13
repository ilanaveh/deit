import json
import os
import matplotlib.pyplot as plt


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def plot_metric_multiple_dirs(models, metric):
    plt.figure(figsize=(10, 6))

    for mdl in models:
        filepath = os.path.join('out', mdl, 'log.txt')
        if os.path.exists(filepath):
            log_data = read_log_file(filepath)
            epochs = [entry['epoch'] for entry in log_data]
            values = [entry[metric] for entry in log_data]
            plt.plot(epochs, values, linestyle='-', label=mdl)
        else:
            print(f"Log file not found in directory: {mdl}")

    plt.title(metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title='Model')
    plt.show()


# Usage example
if __name__ == "__main__":
    models = ['original', 'deit_blur8', 'deit_blur16']
    metric = 'test_acc1'  # Choose: train_loss / test_loss / test_acc1 / test_acc5 / train_lr

    plot_metric_multiple_dirs(models, metric)
