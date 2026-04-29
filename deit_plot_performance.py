import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
from collections import defaultdict
import numpy as np
import re


def read_log_file(filepath):
    with open(filepath, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data


def get_epoch_acc(log_data, epoch, mdl, test_blur):
    for epoch_data in log_data:
        if epoch_data['epoch'] == epoch:
            # if 'deit_blur0-32' in mdl:
            #     if test_blur == 'min':
            #         acc1_key = 'test_acc1'
            #         test_blur_sigma = 0
            #     else:
            #         acc1_key = 'test_blur_max_acc1'
            #         test_blur_sigma = 32
            # # elif ('deit_blur0-16' in mdl) or ('deit_blur16-32' in mdl) or ('deit_blur0-8' in mdl):
            # el
            if re.search(r'deit_blur\d+-\d+', mdl):
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
    'deit_blur0-8_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'original': 'out',
    'deit_blur4': 'out',
    'deit_blur8': 'out',
    'deit_blur16': 'out',
    'deit_blur32': 'out',
    'deit_blur0-32_tmp': 'out',
    'deit_blur4_rep': 'out',
    'deit_blur0_tchr_deit-high-res_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0_tchr_preRes-blur0_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-8_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-16_tchr_deit-high-res_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur0-16_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_RegNetY-160_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_RegNetY-160_hard_rep': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0-16_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0-32_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0-8_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur8_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur16_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur32_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur0-16_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur0-32_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur8_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur32_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur8_tchr_preRes-blur8_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_preRes-blur16_hard': osp.join('out', 'distillation_jobs'),
    'deit_blur8_tchr_preRes-blur8_hard_rep': osp.join('out', 'distillation_jobs'),
    'deit_blur16_tchr_preRes-blur16_hard_rep': osp.join('out', 'distillation_jobs'),
    'deit_blur2_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur4_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur0-2_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur0-4_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur2_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur4_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0-2_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur0-4_tchr_RegNetY-160_hard_BS128': osp.join('out', 'distillation_jobs'),
    'deit_blur8-16_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur16-32_BS128': osp.join('out', 'jobs_after_adding_seed'),
    'deit_blur12_BS128': osp.join('out', 'jobs_after_adding_seed'),
}

# Create a function to get the appropriate color based on model name
# The variable-blur models need to be first (otherwise, the 'blur0' would be considered as the correct key)
color_map = {
    'blur0-32': 'green',
    'blur0-16': 'red',
    'blur8-16': 'brown',
    # 'blur16-32': 'cadetblue',
    'blur16-32': 'olive',
    'blur0-8': 'orange',
    'blur0-8_BS128': 'orange',
    'blur0-2': 'black',
    'blur0-4': 'blue',
    'blur0': 'mediumpurple',
    'original': 'mediumpurple',
    'blur6': 'black',
    'blur8': 'gold',
    'blur16': 'pink',
    'blur32': 'limegreen',
    'blur0_tchr_deit-high-res': 'mediumpurple',  # used to be 'blue' (before adding dashed for teacher)
    'blur0_tchr_RegNetY-160': 'mediumpurple',
    'blur0_tchr_RegNetY-160_BS128': 'mediumpurple',
    'blur0_tchr_preRes-blur0': 'mediumpurple',
    'blur0-8_tchr_RegNetY-160': 'orange',  # used to be 'darkgoldenrod'
    'blur0-8_tchr_RegNetY-160_BS128': 'orange',
    'blur8_tchr_RegNetY-160_BS128': 'gold',
    'blur8_tchr_preRes-blur8': 'gold',
    'blur0-16_tchr_deit-high-res': 'red',  # used to be 'purple'
    'blur0-16_tchr_RegNetY-160': 'red',
    'blur0-16_tchr_RegNetY-160_BS128': 'red',
    'blur16_tchr_RegNetY-160': 'pink',  # used to be 'magenta'
    'blur16_tchr_preRes-blur16': 'pink',
    'blur16_tchr_RegNetY-160_BS128': 'pink',
    'blur32_tchr_RegNetY-160_BS128': 'limegreen',
    'blur0-32_tchr_RegNetY-160_BS128': 'green',
    'blur2': 'grey',
    'blur2_tchr_RegNetY-160_BS128': 'grey',
    'blur0-2_tchr_RegNetY-160_BS128': 'black',
    'blur4': 'cyan',
    'blur4_tchr_RegNetY-160_BS128': 'cyan',
    'blur0-4_tchr_RegNetY-160_BS128': 'blue',
    'blur0-4_tchr_RegNetY-160_BS128': 'blue',
    'blur12': 'blue',
}
# More colors I can add: 'teal', 'navy', 'gold', 'coral', 'indigo', 'turquoise'


def get_color_for_model(model_name):
    if 'tchr' in model_name:
        return color_map[model_name.replace('deit_', '').replace('_hard', '').replace('_rep', '')]
    for blur_level in color_map.keys():
        if blur_level in model_name:
            # if 'BS128' in model_name:
            #     return [np.max([c - .3, 0]) for c in colors.to_rgb(color_map[blur_level])]
            return color_map[blur_level]
    return 'gray'  # Default color if no match is found


def plot_metric(models, metrics, test_blur, ls_dict={}):
    """

    :param models: list of model names (should match the directory names in '/out')
    :param metrics: name of metric to plot / list of four.
    :param test_blur: either 'max' or 'min' - which test-blur to plot (relevant to var_blur models).
    :param ls_dict: keys: substring (in model name), values: linestyle (e.g. '--', ':').

    :return:
    """

    plt.figure(figsize=(10, 6))
    # Initialize a set to keep track of which blur levels have been added to the legend
    legend_added = set()

    for i, metric in enumerate(metrics):
        if len(metrics) > 1:
            plt.subplot(2, 2, i + 1)

        for mdl in models:
            filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
            filepath = filepath if os.path.exists(filepath) else os.path.join('code/Transformers/deit', filepath)
            if os.path.exists(filepath):
                log_data = read_log_file(filepath)
                epochs = [entry['epoch'] for entry in log_data]
                # if ('deit_blur0-32' in mdl) & (test_blur == 'max'):  # if test_blur is 'min', then default is ok.
                #     values = [entry[metric.replace('_', '_blur_max_')] for entry in log_data]
                # if ('deit_blur0-16' in mdl) \
                #         or ('deit_blur16-32' in mdl) \
                #         or ('deit_blur0-32' in mdl)\
                #         or ('deit_blur0-32_rep' in mdl)\
                #         or ('deit_blur0-8' in mdl):
                if re.search(r"deit_blur\d+-\d+", mdl):
                    blur_min = mdl.split('-')[0].split('blur')[1]
                    blur_max = mdl.split('-')[1].split('_')[0]
                    blur2plt = blur_max if (test_blur == 'max') else blur_min if (test_blur == 'min') else -1
                    values = [entry[metric.replace('_', f'_blur_{blur2plt}_')] for entry in log_data]
                else:
                    values = [entry[metric] for entry in log_data]
                color = get_color_for_model(mdl)
                ls = '-'  # default
                lw = 1.5
                for sub_string, line_style in ls_dict.items():
                    if sub_string in mdl:
                        ls = line_style
                        if ls == ':':
                            lw = 2.5
                plt.plot(epochs, values, linestyle=ls, color=color, linewidth=lw)
                if 'original' in mdl:
                    blur_level = 'blur0'
                elif 'tchr' in mdl:
                    blur_level = mdl.strip('deit_')
                else:
                    blur_level = next((blur for blur in color_map.keys() if blur in mdl), None)
                if blur_level and (color not in legend_added):
                    plt.plot([], [], color=color, label=blur_level)  # Add empty plot for legend
                    legend_added.add(color)
            else:
                print(f"Log file not found in directory: {mdl}")

        # Add line styles to legend:
        num2add = 0  # how many entries were added to the legend, beyond 'legend_added' (used for # columns in legend)
        for sub_string, line_style in ls_dict.items():
            if any([sub_string in m for m in models]):
                plt.plot([], [], linestyle=line_style, color='gray', label=sub_string)  # Add empty plot for legend
                num2add += 1

        plt.xlabel('Epoch')
        if metric == 'test_acc1':
            plt.ylabel('Top1 Accuracy')
        plt.grid(True)
        ax = plt.gca()
        ax.set_position([.125, .15, .8, .8])
    nrows = 3
    plt.legend(title='Model', ncol=np.ceil((len(legend_added)+num2add) / nrows))


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

    f, ax = plt.subplots(figsize=(10, 6))

    # Step 1: Group accuracies and model names by color
    color_to_accs = defaultdict(list)
    color_to_names = defaultdict(list)
    color_to_tst_sig = defaultdict(list)

    for mdl in models:
        filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
        filepath = filepath if os.path.exists(filepath) else os.path.join('code/Transformers/deit', filepath)
        if os.path.exists(filepath):
            best_cp_pth = filepath.replace('log.txt', 'best_checkpoint.pth')
            log_data = read_log_file(filepath)
            best_cp = torch.load(best_cp_pth)
            best_epoch = best_cp['epoch']
            acc, test_bl_sig = get_epoch_acc(log_data, best_epoch, mdl, test_blur)
            color = get_color_for_model(mdl)
            if 'tchr' in mdl:
                color += '_tchr'
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
        colors.append(color.replace('_tchr', ''))
        means.append(np.mean(accs))
        stds.append(np.std(accs) if (len(accs) > 1) else np.nan)
        group_label = get_gen_mdl_name(color_to_names[color])
        # lbl = (group_label if group_label else "blur0") if 'tchr' not in color else (group_label+'+Tchr')  # One of the blur0 models is named 'original', so no common prefix would be found.
        lbl = group_label if group_label else "blur0"  # One of the blur0 models is named 'original', so no common prefix would be found.
        x_labels.append(lbl)
        tst_sigmas.append(color_to_tst_sig[color])

    # Step 3: Plot the bars with error bars
    x = np.arange(len(means))
    if np.any([not np.isnan(s) for s in stds]):
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, zorder=3)
    else:
        bars = ax.bar(x, means, color=colors, capsize=5, zorder=3)

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
        'deit_blur0-8_tmp', 'deit_blur0-8_rep', 'deit_blur0-8_tchr_RegNetY-160_hard_BS128',
        'deit_blur16', 'deit_blur16_tmp_new',
        'deit_blur16_tchr_RegNetY-160_hard',  'deit_blur16_tchr_RegNetY-160_hard_rep',
        'deit_blur16_tchr_RegNetY-160_hard_BS128',
        'deit_blur0-16_tmp_fix_bug', 'deit_blur0-16_rep', 'deit_blur0-16_tchr_RegNetY-160_hard',
        'deit_blur0-16_tchr_RegNetY-160_hard_BS128',
        'deit_blur32', 'deit_blur32_tmp_new',
        'deit_blur0-32_tmp', 'deit_blur0-32_tmp_new',

    ]

    models_comp_BS128 = [
        'deit_blur0_tmp_new', 'original',
        'deit_blur0_BS128',
        'deit_blur8', 'deit_blur8_rep',
        'deit_blur8_BS128',
        'deit_blur16', 'deit_blur16_tmp_new',
        'deit_blur16_BS128',
        'deit_blur0-16_tmp_fix_bug', 'deit_blur0-16_rep',
        'deit_blur0-16_BS128'

    ]

    models_all_BS128 = [
        'deit_blur0_BS128', 'deit_blur0_tchr_RegNetY-160_hard_BS128',
        'deit_blur8_BS128',
        'deit_blur0-8_BS128', 'deit_blur0-8_tchr_RegNetY-160_hard_BS128',
        'deit_blur16_BS128', 'deit_blur16_tchr_RegNetY-160_hard_BS128',
        'deit_blur0-16_BS128', 'deit_blur0-16_tchr_RegNetY-160_hard_BS128',
    ]

    models_seed = [
        'deit_blur0_BS128', 'deit_blur0_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur2_BS128', 'deit_blur2_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur0-2_BS128', 'deit_blur0-2_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur4_BS128', 'deit_blur4_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur0-4_BS128', 'deit_blur0-4_tchr_RegNetY-160_hard_BS128',
        'deit_blur8_BS128', 'deit_blur8_tchr_RegNetY-160_hard_BS128',
        'deit_blur0-8_BS128', 'deit_blur0-8_tchr_RegNetY-160_hard_BS128',
        'deit_blur12_BS128',
        'deit_blur16_BS128', 'deit_blur16_tchr_RegNetY-160_hard_BS128',
        'deit_blur8-16_BS128',
        'deit_blur0-16_BS128', 'deit_blur0-16_tchr_RegNetY-160_hard_BS128',
        'deit_blur32_BS128', 'deit_blur32_tchr_RegNetY-160_hard_BS128',
        'deit_blur16-32_BS128',
        'deit_blur0-32_BS128', 'deit_blur0-32_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur0_tchr_preRes-blur0_hard',
        # 'deit_blur8_tchr_preRes-blur8_hard', 'deit_blur16_tchr_preRes-blur16_hard',
        # 'deit_blur8_tchr_preRes-blur8_hard_rep', 'deit_blur16_tchr_preRes-blur16_hard_rep'
    ]
    #
    # models_for_bars = [
    #     'deit_blur0_BS128', 'deit_blur0_tchr_RegNetY-160_hard_BS128',
    #     'deit_blur4_BS128',
    #     'deit_blur0-4_BS128', 'deit_blur0-4_tchr_RegNetY-160_hard_BS128',
    #     'deit_blur8_BS128',
    #     'deit_blur0-8_BS128', 'deit_blur0-8_tchr_RegNetY-160_hard_BS128',
    #     'deit_blur12_BS128',
    #     'deit_blur16_BS128',
    #     'deit_blur0-16_BS128', 'deit_blur0-16_tchr_RegNetY-160_hard_BS128'
    # ]

    models_for_bars = [
        'deit_blur0_BS128',
        'deit_blur8_BS128',
        'deit_blur0-8_BS128'
    ]

    metric = ['test_acc1']  # Choose: train_loss / test_loss / test_acc1 / test_acc5 / train_lr
    # metrics = ['train_loss', 'test_loss', 'train_lr', 'test_acc1']
    # plot_bars(models, test_blur='min')
    plot_bars(models_for_bars, 'max')
    plot_metric(models_seed, metric, 'max', ls_dict={'tchr_RegNet': '--', 'tchr_preRes': ':'})
    plot_metric(models_comp_BS128, metric, 'max', ls_dict={'BS128': '--'})
    plot_metric(models_cleaner_fig, metric, 'max')
    plot_metric(models_all_BS128, metric, 'max')
    plot_bars(models_seed, 'max')
