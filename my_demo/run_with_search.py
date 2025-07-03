# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 16:48
@project:    CRC25
"""
import os
import random
import sys
import pickle
from datetime import time

import numpy as np
import yaml
from pyinstrument import Profiler

from my_demo.config import Config
from my_demo.mip.ModelSolver import ModelSolver
from my_demo.search.SearchSolver import SearchSolver
from my_demo.search.saturated_search.SearchSolverSaturated import SearchSolverSaturated
import gc
import concurrent.futures
import time
sys.path.append("..")
from my_demo.solver.DESolver import DESolver
from visual import visual_line, visual_map, visual_map_foil_modded


# from  utils.common_utils import set_seed

def set_seed(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)


def single_main(route_name=None):
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)

    # 设置地图名
    if route_name is not None:
        config['paths']['route_name'] = route_name

    # 初始化DataLoader，传入base_dir
    config = Config(config, base_dir=base_dir)
    config.out_path = os.path.join(base_dir, "my_demo", "output", "new_search_test")
    # solver = SearchSolver(config)
    solver = SearchSolverSaturated(config)
    solver.do_solve()

    solver.process_solution_from_model()

    visual_data = solver.process_visual_data()
    # # 保存为pickle 方便可视化调试
    # visual_pkl_path = os.path.join(base_dir, "my_demo", "output", "visual", "visual_data.pkl")
    # with open(visual_pkl_path, "wb") as f:
    #     pickle.dump(visual_data, f)

    visual_map_foil_modded(visual_data, config.out_path, config.route_name)
    print(f"{route_name or config.route_name} DONE")

def batch_main():
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)
    routes_dir = os.path.join(base_dir, "data", "train", "routes")
    output_dir = os.path.join(base_dir, "my_demo", "output", "search_test")
    os.makedirs(output_dir, exist_ok=True)
    route_names = [d for d in os.listdir(routes_dir) if os.path.isdir(os.path.join(routes_dir, d))]

    for route_name in route_names:
        try:
            print(f"{route_name} START")
            single_main(route_name)
        except Exception as e:
            print(f"Route {route_name} ERROR: {e}")
    print("ALL DONE")

if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    start_time = time.time()
    # batch_main()
    single_main()
    print("ALL DONE cost time:", time.time() - start_time)
    profiler.stop()
    profiler.write_html("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/profiler.html",show_all=True)
