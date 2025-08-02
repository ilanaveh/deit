"""
02/08/2025
run script deit_eval_performance.py, with different models and test-blurs.
"""
import numpy as np
import subprocess
# from deit_eval_performance import main as eval_main

# models = [
#     'deit_blur0_tmp_new',
#     'deit_blur16_tmp_new',
#     'deit_blur32_tmp_new',
#     'deit_blur0-16_tmp_fix_bug',
#     'deit_blur0-32_tmp_new'
# ]
models = [
    'original',
    'deit_blur16',
    'deit_blur32',
    'deit_blur0-16_rep',
    'deit_blur0-32_tmp'
]

model_dirs = [                                  # within 'deit'
    'out',
    'out',
    'out',
    'out/jobs_from_scratch_main_tmp_code',
    'out'
]


test_blurs = [0, 16, 32]


for mdl, model_dir in zip(models, model_dirs):
    for bl in test_blurs:
        subprocess.run(['python', 'deit_eval_performance.py',
                        '--deit_model_dir', model_dir, '--deit_model_name', mdl, '--blur', str(bl)])

print('done')
