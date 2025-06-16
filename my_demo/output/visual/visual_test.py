# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/11 17:21
@project:    CRC25
"""
import pickle
from my_demo.visual import visual_map_explore,visual_map_foil_modded
import os

base_dir = "//my_demo"
with open(f"{base_dir}/output/visual/visual_data.pkl", 'rb') as f:
    info = pickle.load(f)

visual_map_foil_modded(info, os.path.join(base_dir, "output", "visual"))
