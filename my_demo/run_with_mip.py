# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 10:20
@project:    CRC25
"""
import copy
import itertools
import os
import random
import sys
import pickle
import time

import numpy as np
import pandas as pd
import yaml
from geopandas import GeoDataFrame
from pyinstrument import Profiler
from my_demo.config import Config
from my_demo.mip.LpModelSolver import LpModelSolver
from my_demo.mip.MipDataHolder import MipDataHolder
from my_demo.mip.ModelSolver import ModelSolver
from my_demo.mip.PulpModelSolver import PulpModelSolver
import gc
import concurrent.futures

from my_demo.mip.ScipyModelSolver import ScipyModelSolver

sys.path.append("..")
from my_demo.solver.DESolver import DESolver
from visual import visual_line, visual_map, visual_map_foil_modded


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
    solver = PulpModelSolver(config_obj)
    solver.init_model()
    solver.solve_model()
    solver.process_solution_from_model()
    visual_data = solver.process_visual_data()
    # 输出文件名带route名
    visual_map_foil_modded(visual_data, output_dir, route_name)

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


def batch_run_compare():
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)

    demo_list = []
    dist_diff_list = []
    arc_diff_mod_count_list = []
    arc_diff_rec_count_list = []
    conclusion_list = []
    for i, j in itertools.product(range(5), range(1, 6)):
        # for i,j in [(0,3)]:
        config_dict['paths']['route_name'] = f'osdpm_{i}_{j}'

        # 初始化DataLoader，传入base_dir
        config = Config(config_dict, base_dir=base_dir)

        # solver = PulpModelSolver(config)
        solver = ScipyModelSolver(config)
        # solver = GrbModelSolver(config)
        solver.init_model()
        solver.solve_model()
        with open(f'{base_dir}/solver.pk', 'wb') as file:
            pickle.dump([solver, config], file, protocol=pickle.HIGHEST_PROTOCOL)
        # solver_copy = deep_copy_serialization(solver)
        solver.process_solution_from_model()

        visual_data = solver.process_visual_data()
        visual_map_foil_modded(visual_data, os.path.join(base_dir, "my_demo", "output", "visual_batch"),
                               config.route_name)
        # record the difference between two best route and foil route

        arc_diff_mod_count = len(set(solver.df_best_route['arc']) - set(solver.df_path_foil['arc']))
        arc_diff_foil_count = len(set(solver.df_path_foil['arc']) - set(solver.df_best_route['arc']))
        arc_diff_mod_count_list.append(arc_diff_mod_count)
        arc_diff_rec_count_list.append(arc_diff_foil_count)
        conclusion_list.append(arc_diff_mod_count == 0 and arc_diff_foil_count == 0)
        demo_list.append(f'osdpm_{i}_{j}')
    conclusion_df = pd.DataFrame({'demo': demo_list, 'arc_diff_mod_count': arc_diff_mod_count_list,
                                  'arc_diff_rec_count': arc_diff_rec_count_list, 'overlap': conclusion_list})
    conclusion_df.to_csv(os.path.join(base_dir, "my_demo", "output", "visual_batch", "compare99999.csv"))


def single_main():
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)

    # 初始化DataLoader，传入base_dir
    start = time.time()
    config = Config(config, base_dir=base_dir)
    init_data_end = time.time()
    print(f"Init time: {init_data_end - start}")

    # solver = PulpModelSolver(config)
    solver = ScipyModelSolver(config)
    # solver = ModelSolver(config)
    solver.init_model()
    init_model_end = time.time()
    print(f"Init model time: {init_model_end - init_data_end}")
    solver.solve_model()
    solve_end = time.time()
    print(f"Solve time: {solve_end - init_model_end}")
    with open(f'{base_dir}/solver.pk', 'wb') as file:
        pickle.dump([solver, config], file, protocol=pickle.HIGHEST_PROTOCOL)
    # solver_copy = deep_copy_serialization(solver)
    solver.process_solution_from_model()

    visual_data = solver.process_visual_data()
    # # 保存为pickle 方便可视化调试
    # visual_pkl_path = os.path.join(base_dir, "my_demo", "output", "visual", "visual_data.pkl")
    # with open(visual_pkl_path, "wb") as f:
    #     pickle.dump(visual_data, f)

    # visual_map_explore(visual_data, os.path.join(base_dir, "my_demo", "output", "visual"))
    visual_map_foil_modded(visual_data, os.path.join(base_dir, "my_demo", "output", "visual_batch"), config.route_name)
    # varify_df = modify_recovery_varify(solver,config,base_dir)
    # varify_df.to_csv(os.path.join(base_dir, "my_demo", "output", "visual_batch", "recovery_varify.csv"))
    print("DONE")


def modify_recovery_varify(solver_modified, config, base_dir):
    # 修改后的arc
    modified_map = solver_modified.org_map_df
    modified_arc = modified_map[modified_map['modified'].str.len().gt(0)]
    modified_arc_list = modified_arc['arc'].tolist()
    # 修改后的最优路由
    modified_best_route = solver_modified.df_best_route
    dist_diff_list = []
    arc_diff_mod_count_list = []
    arc_diff_rec_count_list = []
    conclusion_list = []
    for rec_arc in modified_arc_list:
        # solver_rec = deep_copy_serialization(solver_copy)

        with open(f'{base_dir}/solver.pk', 'rb') as file:
            solver_rec, config = pickle.load(file)
        # 重新初始化data_holder
        # data_holder中的数据都是静态属性,需要重新赋值
        solver_rec.load_basic_data()
        solver_rec.data_process()

        solver_rec.process_solution_from_model([rec_arc])
        visual_data = solver_rec.process_visual_data()
        visual_map_foil_modded(visual_data, os.path.join(base_dir, "my_demo", "output", "visual_batch"),
                               config.route_name + f'_rec_{rec_arc}', [rec_arc])
        recovery_best_route = solver_rec.df_best_route
        # 对比best route
        dist_diff = modified_best_route['my_weight'].sum() - recovery_best_route['my_weight'].sum()
        arc_diff_mod_count = len(set(modified_best_route['arc']) - set(recovery_best_route['arc']))
        arc_diff_rec_count = len(set(recovery_best_route['arc']) - set(modified_best_route['arc']))
        if arc_diff_mod_count == 0 and arc_diff_rec_count == 0:
            conclusion = '无效修改'
            print(f'修改边{rec_arc}是无效操作,不影响最短路')
        else:
            conclusion = '有效修改'
            print(f'修改边{rec_arc}影响{arc_diff_mod_count + arc_diff_rec_count}条边')
        dist_diff_list.append(dist_diff)
        arc_diff_mod_count_list.append(arc_diff_mod_count)
        arc_diff_rec_count_list.append(arc_diff_rec_count)
        conclusion_list.append(conclusion)
    # 结论df
    conclusion_df = pd.DataFrame({
        'arc': modified_arc_list,
        'modified': modified_arc['modified'],
        'weight_diff': dist_diff_list,
        'arc_diff_mod_count': arc_diff_mod_count_list,
        'arc_diff_rec_count': arc_diff_rec_count_list,
        'conclusion': conclusion_list
    })
    return conclusion_df


def deep_copy_serialization(obj):
    return pickle.loads(pickle.dumps(obj))


def solve_mip(config) -> GeoDataFrame:
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "..")
    base_dir = os.path.abspath(base_dir)

    # 初始化DataLoader，传入base_dir
    config = Config(config, base_dir=base_dir)

    solver = ScipyModelSolver(config)
    solver.init_model()
    solver.solve_model()
    with open(f'{base_dir}/solver.pk', 'wb') as file:
        pickle.dump([solver, config], file, protocol=pickle.HIGHEST_PROTOCOL)
    solver.process_solution_from_model()

    modified_arcs = solver.org_map_df[solver.org_map_df['modified'].str.len().gt(0)]
    return modified_arcs


# if __name__ == "__main__":
#     # profiler = Profiler()
#     # profiler.start()
#     # batch_main()
#     # batch_run_compare()
#     single_main()
#     # profiler.stop()
#     # profiler.write_html("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/profiler.html",show_all=True)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    modified_arcs = solve_mip(config)
    print(modified_arcs)
