"""
02/08/2025
run script deit_eval_performance.py, with different models and test-blurs.

25/2/26
Add option to write results to file.
"""
import subprocess
from datetime import date
import os.path as osp


write2file = True
write_date = False  # relevant only if write2file = True

out_pth = '/home/projects/bagon/ilanaveh/code/Transformers/deit/out_from_eval_performance/deit_eval_results.txt'
# from deit_eval_performance import main as eval_main

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
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3/8/25 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# models = [
#     'deit_blur0_tmp_new',
#     'deit_blur16_tmp_new',
#     'deit_blur32_tmp_new',
#     'deit_blur0-16_tmp_fix_bug',
#     'deit_blur0-32_tmp_new'
# ]

# models = [
#     'original',
#     'deit_blur16',
#     'deit_blur32',
#     'deit_blur0-16_rep',
#     'deit_blur0-32_tmp'
# ]
#
# model_dirs = [                                  # within 'deit'
#     'out',
#     'out',
#     'out',
#     'out/jobs_from_scratch_main_tmp_code',
#     'out'
# ]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~` 25/2/26 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Taken model list from 'deit_plot_performance.py' ('models_seed').
models = [
        # 'deit_blur0_BS128',
        'deit_blur0_tchr_RegNetY-160_hard_BS128',
        ## 'deit_blur2_BS128', 'deit_blur2_tchr_RegNetY-160_hard_BS128',
        ## 'deit_blur0-2_BS128', 'deit_blur0-2_tchr_RegNetY-160_hard_BS128',
        ## 'deit_blur4_BS128', 'deit_blur4_tchr_RegNetY-160_hard_BS128',
        ## 'deit_blur0-4_BS128', 'deit_blur0-4_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur8_BS128',
        'deit_blur8_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur0-8_BS128',
        'deit_blur0-8_tchr_RegNetY-160_hard_BS128',
        # 'deit_blur16_BS128',
        'deit_blur16_tchr_RegNetY-160_hard_BS128',
        'deit_blur8-16_BS128',
        'deit_blur0-16_BS128', 'deit_blur0-16_tchr_RegNetY-160_hard_BS128',
        'deit_blur32_BS128', 'deit_blur32_tchr_RegNetY-160_hard_BS128',
        'deit_blur16-32_BS128',
        'deit_blur0-32_BS128', 'deit_blur0-32_tchr_RegNetY-160_hard_BS128',
        ## 'deit_blur0_tchr_preRes-blur0_hard',
        ## 'deit_blur8_tchr_preRes-blur8_hard', 'deit_blur16_tchr_preRes-blur16_hard',
        ## 'deit_blur8_tchr_preRes-blur8_hard_rep', 'deit_blur16_tchr_preRes-blur16_hard_rep'
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

test_blurs = [0, 8, 16, 32]

if write2file and write_date:
    with open(out_pth, "a", encoding="utf-8") as f:
        f.write(f"{date.today()}\n")

for mdl in models:
    model_dir = model_out_dict[mdl]

    model_type = 'deit_base_distilled_patch16_224' if 'distillation' in model_dir else 'deit_base_patch16_224'

    if write2file:
        with open(out_pth, "a", encoding="utf-8") as f:
            f.write(f"\nmodel: {model_dir}/{mdl}\n")

    for bl in test_blurs:
        if write2file:
            subprocess.run(['python', 'deit_eval_performance.py', '--write2file', '--out_pth', out_pth,
                            '--model', model_type, '--deit_model_dir', model_dir, '--deit_model_name', mdl,
                            '--blur', str(bl)])
        else:
            subprocess.run(['python', 'deit_eval_performance.py', '--model', model_type,
                            '--deit_model_dir', model_dir, '--deit_model_name', mdl, '--blur', str(bl)])

if write2file:
    with open(out_pth, "a", encoding="utf-8") as f:
        f.write("=" * 60)
        f.write("\n")
print('done')
