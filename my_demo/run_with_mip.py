# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 10:20
@project:    CRC25
"""
import os
import random
import sys
import pickle

import numpy as np
import yaml
from pyinstrument import Profiler
from my_demo.config import Config
from my_demo.mip.MipModelSolver import ModelSolver
from my_demo.mip.MipModelSolverNew import ModelSolverNew

sys.path.append("..")
from my_demo.solver.DESolver import DESolver
from visual import visual_line, visual_map, visual_map_explore,visual_map_foil_modded


# from  utils.common_utils import set_seed

def set_seed(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)


def main():
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)

    # 初始化DataLoader，传入base_dir
    config = Config(config, base_dir=base_dir)

    solver = ModelSolverNew(config)
    # solver = ModelSolver(config)
    solver.init_model()
    solver.solve_model()
    solver.process_solution_from_model()

    visual_data = solver.process_visual_data()
    # 保存为pickle 方便可视化调试
    visual_pkl_path = os.path.join(base_dir, "my_demo", "output", "visual", "visual_data.pkl")
    with open(visual_pkl_path, "wb") as f:
        pickle.dump(visual_data, f)

    visual_map_explore(visual_data, os.path.join(base_dir, "my_demo", "output", "visual"))
    visual_map_foil_modded(visual_data, os.path.join(base_dir, "my_demo", "output", "visual"))
    print("DONE")


if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # profiler.write_html("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/profiler.html",show_all=True)
