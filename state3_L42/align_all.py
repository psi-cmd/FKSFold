#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymol import cmd
import re
from time import sleep

def get_diffusion_idx_and_particle_idx(obj_name: str) -> tuple[int, int]:
    match = re.match(r"center_(\d*?)_macro_(\d*?)", obj_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# 遍历当前会话中所有对象名
for obj in cmd.get_object_list():
    # diff_idx, particle_idx = get_diffusion_idx_and_particle_idx(obj)
    # 跳过 "8PPZ" 本身
    if obj == "center_7_macro_2":
        continue
    # 执行对齐：将 obj（mobile）对齐到 8PPZ（target）
    cmd.align(obj, "center_7_macro_2 and chain A")
    sleep(1)
