import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import numpy as np


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def get_epoch_acc(log_data, epoch, mdl, test_blur):
    for epoch_data in log_data:
        if epoch_data['epoch'] == epoch:
            if 'deit_blur0-32' in mdl:
                if test_blur == 'min':
                    acc1_key = 'test_acc1'
                    test_blur_sigma = 0
                else:
                    acc1_key = 'test_blur_max_acc1'
                    test_blur_sigma = 32
            elif ('deit_blur0-16' in mdl) or ('deit_blur16-32' in mdl) or ('deit_blur0-8' in mdl):
                blur_min = mdl.split('-')[0].split('blur')[1]
                blur_max = mdl.split('-')[1].split('_')[0]
                blur2plt = blur_max if (test_blur == 'max') else blur_min if (test_blur == 'min') else -1
                acc1_key = f'test_blur_{blur2plt}_acc1'

                test_blur_sigma = blur2plt
            else:

                acc1_key = 'test_acc1'
                # all models have 'blur' in name, except for 'original'
                test_blur_sigma = mdl.split('blur')[1].split('_')[0] if ('blur' in mdl) else 0

            return epoch_data[acc1_key], test_blur_sigma


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
    'deit_blur0-16_tmp_fix_bug': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16-32_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-16_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-32_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16-32_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-8_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-8_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'original': 'out',
    'deit_blur4': 'out',
    'deit_blur8': 'out',
    'deit_blur16': 'out',
    'deit_blur32': 'out',
    'deit_blur0-32_tmp': 'out',
    'deit_blur4_rep': 'out',
    'deit_blur0_tchr_deit-high-res_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-8_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-16_tchr_deit-high-res_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-16_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_RegNetY-160_hard_rep': osp.join('out', 'distillation_jobs')
}

# Create a function to get the appropriate color based on model name
color_map = {
    'blur0-32': 'green',
    'blur0-16': 'magenta',
    'blur16-32': 'olive',
    'blur0-8': 'orange',
    'blur0': 'cyan',
    'original': 'cyan',
    'blur2': 'green',
    'blur4': 'red',
    'blur6': 'black',
    'blur8': 'gold',
    'blur16': 'pink',
    'blur32': 'limegreen',
    'blur0_tchr_deit-high-res': 'blue',
    'blur0_tchr_RegNetY-160': 'blue',
    'blur0-8_tchr_RegNetY-160': 'darkgoldenrod',
    'blur0-16_tchr_deit-high-res': 'red',
    'blur0-16_tchr_RegNetY-160': 'red',
    'blur16_tchr_RegNetY-160': 'purple'
}
# More colors I can add: 'teal', 'navy', 'gold', 'coral', 'indigo', 'turquoise'


def get_color_for_model(model_name):
    if 'tchr' in model_name:
        return color_map[model_name.replace('deit_', '').replace('_hard', '').replace('_rep', '')]
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
                elif ('deit_blur0-16' in mdl) \
                        or ('deit_blur16-32' in mdl) \
                        or ('deit_blur0-32_rep' in mdl)\
                        or ('deit_blur0-8' in mdl):
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
                elif 'tchr' in mdl:
                    blur_level = mdl.strip('deit_')
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


def get_gen_mdl_name(strings):
    """Get general model name (format: blurX or blurX-Y), from a list of model names with common blur"""
    if not strings:
        return ""
    prefix = os.path.commonprefix(strings)  # the common part in all list items

    # Get only the 'blurX' part:
    split_prefix = np.array(prefix.split('_'))
    blur_ind = ['blur' in x for x in split_prefix]
    if any(blur_ind):
        if 'tchr' in split_prefix:
            return split_prefix[blur_ind][0] + '_tchr'
        return split_prefix[blur_ind][0]
    else:
        # Get the blur from one of the individual models:
        for string in strings:
            split_string = np.array(string.split('_'))
            blur_ind = ['blur' in x for x in split_string]
            if any(blur_ind):
                return split_string[blur_ind][0]

    return 'unknown'


def plot_bars(models, test_blur):
    """
    Plot accuracy of the best checkpoint.
    If there was more than one repetition, each bar represents the mean & the error bars - the std.
    """

    f, ax = plt.subplots(figsize=(5, 3))

    # Step 1: Group accuracies and model names by color
    color_to_accs = defaultdict(list)
    color_to_names = defaultdict(list)
    color_to_tst_sig = defaultdict(list)

    for mdl in models:
        filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
        best_cp_pth = os.path.join(model_out_dict[mdl], mdl, 'best_checkpoint.pth')
        if os.path.exists(filepath):
            log_data = read_log_file(filepath)
            best_cp = torch.load(best_cp_pth)
            best_epoch = best_cp['epoch']
            acc, test_bl_sig = get_epoch_acc(log_data, best_epoch, mdl, test_blur)
            color = get_color_for_model(mdl)
            color_to_accs[color].append(acc)
            color_to_names[color].append(mdl)
            color_to_tst_sig[color] = test_bl_sig
        else:
            print(f"Log file not found in directory: {mdl}")

    # Step 2: Prepare data for plotting
    colors = []
    means = []
    stds = []
    x_labels = []
    tst_sigmas = []

    for color, accs in color_to_accs.items():
        colors.append(color)
        means.append(np.mean(accs))
        stds.append(np.std(accs) if (len(accs) > 1) else np.nan)
        group_label = get_gen_mdl_name(color_to_names[color])
        x_labels.append(group_label if group_label else "blur0")  # One of the blur0 models is named 'original', so no common prefix would be found.
        tst_sigmas.append(color_to_tst_sig[color])

    # Step 3: Plot the bars with error bars
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, zorder=3)

    # Add labels:
    ax.bar_label(bars, labels=[f"{m:.1f}" for m in means], padding=3, fontsize=9)

    # Add a table at the bottom of the Axes
    # 1. for 1st row (train blur), remove 'blur' from x_labels:
    train_blurs_for_tbl = [x.split('blur')[1] for x in x_labels]

    plt.ylim([0, 100])
    plt.xlim([-.5, len(x) - .5])
    plt.ylabel("Top-1 Accuracy")
    plt.xticks([])
    plt.grid(axis='y', zorder=0)
    plt.title('Performance on ' + f'{test_blur}imal'.upper() + ' blur-level in range')

    the_table = plt.table(cellText=[train_blurs_for_tbl, tst_sigmas],
                          rowLabels=['Train Blur', 'Test Blur'],
                          colLabels=['' for x in train_blurs_for_tbl],
                          loc='bottom',
                          cellLoc='center',
                          fontsize=20)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # List of all models except for 'deit_blur8_tmp_new' (saved in 'jobs_from_scratch_main_tmp_code'), since there was
    # something wrong with its training.
    models = [
        # models saved in 'out/jobs_from_scratch_main_tmp_code':
        'deit_blur0_tmp_new', 'deit_blur2_tmp_new', 'deit_blur4_tmp_new', 'deit_blur6_tmp_new', 'deit_blur6_rep',
        'deit_blur8_rep', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
        'deit_blur0-16_tmp_fix_bug',  # this is instead 'deit_blur0-16_tmp' which stopped before training ended (5/5/25)
        'deit_blur0-32_tmp_new', 'deit_blur16-32_tmp', 'deit_blur0-8_tmp',
        # Repetitions of the RandBlur jobs:
        'deit_blur0-16_rep', 'deit_blur16-32_rep', 'deit_blur0-8_rep',
        # models saved in 'out':
        'original', 'deit_blur4', 'deit_blur8', 'deit_blur16', 'deit_blur32', 'deit_blur0-32_tmp', 'deit_blur4_rep',
        # models saved in 'distillation_jobs':
        'deit_blur0_tchr_deit-high-res_hard', 'deit_blur0-16_tchr_deit-high-res_hard',
        'deit_blur0_tchr_RegNetY-160_hard', 'deit_blur0-16_tchr_RegNetY-160_hard'
    ]

    models_cleaner_fig = [
        'deit_blur0_tmp_new', 'original', 'deit_blur0_tchr_RegNetY-160_hard',
        'deit_blur8', 'deit_blur8_rep',
        'deit_blur0-8_tmp', 'deit_blur0-8_rep',
        'deit_blur16', 'deit_blur16_tmp_new',
        'deit_blur16_tchr_RegNetY-160_hard',  'deit_blur16_tchr_RegNetY-160_hard_rep',
        'deit_blur0-16_tmp_fix_bug', 'deit_blur0-16_rep', 'deit_blur0-16_tchr_RegNetY-160_hard',
        'deit_blur32', 'deit_blur32_tmp_new',
        'deit_blur0-32_tmp', 'deit_blur0-32_tmp_new',

    ]

    metric = ['test_acc1']  # Choose: train_loss / test_loss / test_acc1 / test_acc5 / train_lr
    # metrics = ['train_loss', 'test_loss', 'train_lr', 'test_acc1']
    # plot_bars(models, test_blur='min')
    plot_metric(models_cleaner_fig, metric, 'max')
