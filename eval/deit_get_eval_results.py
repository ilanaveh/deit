"""
27/4/26
Code for creating results dict, with structure: {mdl_name: {repX: {test_X: Value}}} - similar to results of apvit
(organized in APViT/eval/deit_get_eval_results.py).

Take code from deit_plot_performance.py & run_deit_eval_performance.py.
"""
import os
import json
import torch
import re

test_blurs = [0, 8]

models = [
        'deit_blur0_BS128',
        'deit_blur8_BS128',
        'deit_blur0-8_BS128'
    ]

model_out_dict = {
    'deit_blur0_BS128': os.path.join('out', 'jobs_after_adding_seed'),
    'deit_blur8_BS128': os.path.join('out', 'jobs_after_adding_seed'),
    'deit_blur0-8_BS128': os.path.join('out', 'jobs_after_adding_seed'),

}


def get_epoch_acc(log_data, epoch, mdl, test_blur):
    for epoch_data in log_data:
        if epoch_data['epoch'] == epoch:
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


for mdl in models:
    filepath = os.path.join(model_out_dict[mdl], mdl, 'log.txt')
    filepath = filepath if os.path.exists(filepath) else os.path.join('code/Transformers/deit', filepath)
    if os.path.exists(filepath):
        best_cp_pth = filepath.replace('log.txt', 'best_checkpoint.pth')
        with open(filepath, 'r') as file:
            log_data = [json.loads(line) for line in file]
        best_cp = torch.load(best_cp_pth)
        best_epoch = best_cp['epoch']
        acc, test_bl_sig = get_epoch_acc(log_data, best_epoch, mdl, test_blur)
    else:
        print(f"Log file not found in directory: {mdl}")

