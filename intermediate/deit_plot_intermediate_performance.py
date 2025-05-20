"""
7/5/25
Code for plotting the best-performance of each downstream-training model.
"""

import os


out_dir = 'intermediate/out/new'


def plot_metric(models, metric):
        for mdl in models:
            filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
            if os.path.exists(filepath):
                log_data = read_log_file(filepath)
                epochs = [entry['epoch'] for entry in log_data]
                if ('deit_blur0-32' in mdl) & (test_blur == 'max'):  # if test_blur is 'min', then default is ok.
                    values = [entry[metric.replace('_', '_blur_max_')] for entry in log_data]
                elif ('deit_blur0-16' in mdl) or ('deit_blur16-32' in mdl) or ('deit_blur0-32_rep' in mdl):
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