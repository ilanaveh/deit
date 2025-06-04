"""
3/6/25
For future usages: save the "class_to_idx" dictionary of imagenet.
"""
import json
import os.path as osp


# I obtained the dictionary, by putting a breakpoint at:
# /usr/local/lib/python3.10/dist-packages/torchvision/datasets/folder.py:146
# (after the line: "classes, class_to_idx = self.find_classes(self.root)")
# and running the 'main_tmp.py' code.

save_dir = 'out_from_save_class_to_idx'

with open(osp.join(save_dir, 'class_to_idx.txt'), 'w') as file:
    file.write(json.dumps(class_to_idx))  # use `json.loads` to do the reverse



