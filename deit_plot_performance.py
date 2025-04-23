import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def get_epoch_acc(log_data, epoch):
    for epoch_data in log_data:
        if epoch_data['epoch'] == epoch:
            return epoch_data['test_acc1']


# Dictionary for 'out' folder of each model:
model_out_dict = {
    'deit_blur0_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur2_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur4_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur6_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur8_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur32_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-32_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur6_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur8_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-16_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16-32_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'original': 'out',
    'deit_blur4': 'out',
    'deit_blur8': 'out',
    'deit_blur16': 'out',
    'deit_blur32': 'out',
    'deit_blur0-32_tmp': 'out',
    'deit_blur4_rep': 'out'
}

# Create a function to get the appropriate color based on model name
color_map = {
    'blur0-32': 'cyan',
    'blur0-16': 'pink',
    'blur16-32': 'yellow',
    'blur0': 'blue',
    'original': 'blue',
    'blur2': 'green',
    'blur4': 'red',
    'blur6': 'black',
    'blur8': 'purple',
    'blur16': 'orange',
    'blur32': 'brown'

}


def get_color_for_model(model_name):
    for blur_level in color_map.keys():
        if blur_level in model_name:
            return color_map[blur_level]
    return 'gray'  # Default color if no match is found


def plot_metric(models, metrics, test_blur):
    """

    :param models: list of model names (should match the directory names in '/out')
    :param metrics: name of metric to plot / list of four.
    :param test_blur: either 'max' or 'min' - which test-blur to plot (relevant to var_blur models).

    :return:
    """

    plt.figure(figsize=(4.5, 3))
    # Initialize a set to keep track of which blur levels have been added to the legend
    legend_added = set()

    for i, metric in enumerate(metrics):
        if len(metrics) > 1:
            plt.subplot(2, 2, i + 1)

        for mdl in models:
            filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
            if os.path.exists(filepath):
                log_data = read_log_file(filepath)
                epochs = [entry['epoch'] for entry in log_data]
                if ('deit_blur0-32' in mdl) & (test_blur == 'max'):  # if test_blur is 'min', then default is ok.
                    values = [entry[metric.replace('_', '_blur_max_')] for entry in log_data]
                elif ('deit_blur0-16' in mdl) or ('deit_blur16-32' in mdl):
                    blur_min = mdl.split('-')[0].split('blur')[1]
                    blur_max = mdl.split('-')[1].split('_')[0]
                    blur2plt = blur_max if (test_blur == 'max') else blur_min if (test_blur == 'min') else -1
                    values = [entry[metric.replace('_', f'_blur_{blur2plt}_')] for entry in log_data]
                else:
                    values = [entry[metric] for entry in log_data]
                color = get_color_for_model(mdl)
                plt.plot(epochs, values, linestyle='-', color=color)
                if 'original' in mdl:
                    blur_level = 'blur0'
                else:
                    blur_level = next((blur for blur in color_map.keys() if blur in mdl), None)
                if blur_level and (blur_level not in legend_added):
                    plt.plot([], [], color=color, label=blur_level)  # Add empty plot for legend
                    legend_added.add(blur_level)
            else:
                print(f"Log file not found in directory: {mdl}")

        plt.xlabel('Epoch')
        if metric == 'test_acc1':
            plt.ylabel('Top1 Accuracy')
        plt.grid(True)
        ax = plt.gca()
        ax.set_position([.125, .15, .8, .8])

    plt.legend(title='Model')


def plot_bars(models):
    """
    plot accuracy of the best checkpoint of each model (not entire progression during training as in 'plot_metric').
    """

    f = plt.figure(figsize=(4.5, 2.6))

    best_accs = {mdl: 0 for mdl in models}

    for mdl in models:
        filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
        best_cp_pth = os.path.join(model_out_dict[mdl], mdl, 'best_checkpoint.pth')
        if os.path.exists(filepath):
            log_data = read_log_file(filepath)
            best_cp = torch.load(best_cp_pth)
            best_epoch = best_cp['epoch']
            best_accs[mdl] = get_epoch_acc(log_data, best_epoch)
        else:
            print(f"Log file not found in directory: {mdl}")

    bars = plt.bar(best_accs.keys(), best_accs.values(), zorder=3)
    for i, bar in enumerate(bars):
        bar.set_color(plt.rcParams['axes.prop_cycle'].by_key()['color'][
                          i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])])

    plt.title('Validation Performance')
    plt.ylabel('Top1 Accuracy')
    plt.grid(axis='y', zorder=0)
    plt.ylim([0, 100])


if __name__ == "__main__":
    # , 'deit_8gpu', 'deit_blur4', 'deit_blur32'
    # models = ['original', 'deit_blur4', 'deit_blur4_rep', 'deit_blur8', 'deit_blur16', 'deit_blur32', 'deit_blur0-32',
    #           'deit_blur0-32_tmp', 'deit_blur0_rep']
    # , 'deit_blur0-32_rep'

    models = [
        # models saved in 'out/jobs_from_scratch_main_tmp_code':
        'deit_blur0_tmp_new', 'deit_blur2_tmp_new', 'deit_blur4_tmp_new', 'deit_blur6_tmp_new', 'deit_blur8_tmp_new',
        'deit_blur32_tmp_new', 'deit_blur0-32_tmp_new', 'deit_blur6_rep', 'deit_blur8_rep', 'deit_blur16_tmp_new',
        'deit_blur0-16_tmp', 'deit_blur16-32_tmp',
        # models saved in 'out':
        'original', 'deit_blur4', 'deit_blur8', 'deit_blur16', 'deit_blur32', 'deit_blur0-32_tmp', 'deit_blur4_rep']
    # models = [
    #     ['deit_blur0_tmp_new', 'deit_blur4_tmp_new', 'deit_blur8_rep',
    #      'deit_blur32_tmp_new', 'deit_blur0-32_tmp_new', 'deit_blur16_tmp_new'],
    #     []]
    metric = ['test_acc1']  # Choose: train_loss / test_loss / test_acc1 / test_acc5 / train_lr
    # metrics = ['train_loss', 'test_loss', 'train_lr', 'test_acc1']
    # plot_bars(models)
    plot_metric(models, metric, 'min')
