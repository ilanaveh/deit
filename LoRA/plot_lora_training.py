"""
03/08/2025
Use the data files downloaded from TB
(tensorboard --logdir code/Transformers/deit/LoRA/board/150_epochs/4_classes or /2_classes),
and plot them manually so I can control the colors.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

save_figs = False

blurs = ['0', '16', '32', '0-16', '0-32']

lora_dir = '/home/projects/bagon/ilanaveh/code/Transformers/deit/LoRA'
tb_data_dir = os.path.join(lora_dir, 'data_from_tb')

types = ['2_classes', '4_classes']
ylims = {'2_classes': [70, 100], '4_classes': [40, 90]}
color_map = {
    'blur0-32': 'cyan',
    'blur0-16': 'orange',
    'blur16-32': 'olive',
    'blur0': 'blue',
    'original': 'blue',
    'blur2': 'green',
    'blur4': 'red',
    'blur6': 'black',
    'blur8': 'purple',
    'blur16': 'pink',
    'blur32': 'brown'

}

file_nm_frmt = 'run-finetune_deit_model_blurXX_lora_blurXX-tag-Accuracy_Val_Acc.csv'

for tp in types:
    plt.figure()
    cur_dir = os.path.join(tb_data_dir, tp)
    for bl in blurs:
        data_file = file_nm_frmt.replace('XX', bl)
        if tp == '4_classes':
            data_file = data_file.replace('-tag', '_4cls-tag')
        cur_data = pd.read_csv(os.path.join(cur_dir, data_file))
        clr = color_map['blur'+bl]
        plt.plot(cur_data['Step'], cur_data['Value'], label=bl, color=clr, linewidth=2)
    plt.title(tp)
    plt.legend()
    plt.ylim(ylims[tp])
    if save_figs:
        plt.savefig(os.path.join(lora_dir, 'from_plot_lora_training', f'{tp}.png'))
plt.show()



print('done')
