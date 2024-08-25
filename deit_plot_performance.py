import json
import os
import matplotlib.pyplot as plt


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def plot_metric(models, metrics):
    """

    :param models: list of model names (should match the directory names in '/out')
    :param metric: name of metric to plot / list of four.
    :param only_one: Whether to plot only one metric, or a grid of four.
    :return:
    """

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        if len(metrics) > 1:
            plt.subplot(2, 2, i + 1)

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


if __name__ == "__main__":
    # , 'deit_8gpu', 'deit_blur4', 'deit_blur32'
    models = ['original', 'deit_blur8', 'deit_blur16', 'deit_blur32']
    metric = ['test_acc1']  # Choose: train_loss / test_loss / test_acc1 / test_acc5 / train_lr
    # metrics = ['train_loss', 'test_loss', 'train_lr', 'test_acc1']
    plot_metric(models, metric)
