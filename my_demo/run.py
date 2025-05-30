import os
import random
import sys

import numpy as np
import yaml
from pyinstrument import Profiler
from my_demo.config import Config

sys.path.append("..")
from my_demo.solver.DESolver import DESolver
from visual import visual_line, visual_map
from  utils.common_utils import set_seed



def main():
    set_seed()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动查找examples目录作为基础路径
    # 假设config.yaml和my_demo在同级，examples与其同级
    base_dir = os.path.join(current_dir, "..", "examples")
    base_dir = os.path.abspath(base_dir)

    # 初始化DataLoader，传入base_dir
    config = Config(config, base_dir=base_dir)

    # 构造args
    args = {
        'basic_network_path': config.basic_network_path,
        'foil_json_path': config.foil_json_path,
        'df_path_foil_path': config.df_path_foil_path,
        'gdf_coords_path': config.gdf_coords_path,
        'heuristic': config.heuristic,
        'heuristic_f': config.heuristic_f,
        'jobs': 10,
        'attrs_variable_names': config.attrs_variable_names,
        "n_perturbation": config.n_perturbation,
        "operator_p": config.operator_p,
        "user_model": config.user_model,
        "meta_map": config.meta_map
    }

    solver = DESolver(config)
    solver.run()
    visual_line(solver)
    visual_map(solver)
    # todo solution
    print("DONE")


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    main()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True,show_all=True))
    profiler.write_html("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/profiler.html",show_all=True)