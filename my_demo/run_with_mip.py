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
import gc
import concurrent.futures
sys.path.append("..")
from my_demo.solver.DESolver import DESolver
from visual import visual_line, visual_map,visual_map_foil_modded


# from  utils.common_utils import set_seed

def set_seed(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)


def run_for_route(route_name, config_template, base_dir, output_dir):
    # 复制config并设置route_name
    import copy
    config = copy.deepcopy(config_template)
    config['paths']['route_name'] = route_name
    # 输出目录
    # route_output_dir = os.path.join(output_dir, route_name)
    os.makedirs(output_dir, exist_ok=True)
    # 初始化Config
    config_obj = Config(config, base_dir=base_dir)
    solver = ModelSolverNew(config_obj)
    solver.init_model()
    solver.solve_model()
    solver.process_solution_from_model()
    visual_data = solver.process_visual_data()
    # 输出文件名带route名
    visual_map_foil_modded(visual_data, output_dir,route_name)

    del solver.model
    del solver
    gc.collect()
    print(f"Route {route_name} DONE, output in {output_dir}")

def run_for_route_wrapper(args):
    # 为多进程包装器，避免lambda导致pickle问题
    route_name, config_template, base_dir, output_dir = args
    try:
        run_for_route(route_name, config_template, base_dir, output_dir)
        return (route_name, "DONE")
    except Exception as e:
        return (route_name, f"Error: {e}")
#todo 不知道为什么pulp跑批量 后面的模型会继承前面模型的约束  改成多进程也不行
def batch_main():
    set_seed()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_template = yaml.safe_load(f)
    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)
    routes_dir = os.path.join(base_dir, "data", "train", "routes")
    output_dir = os.path.join(base_dir, "my_demo", "output", "visual_batch")
    os.makedirs(output_dir, exist_ok=True)
    # 获取所有route目录
    route_names = [d for d in os.listdir(routes_dir) if os.path.isdir(os.path.join(routes_dir, d))]
    args_list = [(route_name, config_template, base_dir, output_dir) for route_name in route_names]
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for route_name, status in executor.map(run_for_route_wrapper, args_list):
            print(f"Route {route_name}: {status}")
            results.append((route_name, status))
    print("ALL DONE")

def single_main():
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
    # # 保存为pickle 方便可视化调试
    # visual_pkl_path = os.path.join(base_dir, "my_demo", "output", "visual", "visual_data.pkl")
    # with open(visual_pkl_path, "wb") as f:
    #     pickle.dump(visual_data, f)

    # visual_map_explore(visual_data, os.path.join(base_dir, "my_demo", "output", "visual"))
    visual_map_foil_modded(visual_data, os.path.join(base_dir, "my_demo", "output", "visual_batch"),config.route_name)
    print("DONE")
if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()
    # batch_main()
    single_main()
    # profiler.stop()
    # profiler.write_html("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/profiler.html",show_all=True)
