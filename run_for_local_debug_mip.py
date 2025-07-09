# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/7/9 10:50
@project:    CRC25
"""
import os
import sys

# 在所有入口文件（如 submission_template.py）顶部添加：
import os
import sys
import yaml

# 获取当前文件所在目录（即submission目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（即submission的父目录）
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # 将项目根目录加入路径

import argparse
import json

from config import Config
from src.solver.MipSolver import MipSolver
from src.calc.common_utils import set_seed

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

base_dir = os.path.abspath(current_dir)
out_path = os.path.join(base_dir, "my_demo", "output", "submission_out")

def store_results(output_path, map_df, op_list):
    map_df_path = os.path.join(output_path, "map_df.gpkg")
    op_list_path = os.path.join(output_path, "op_list.json")

    map_df.to_file(map_df_path, driver='GPKG')
    with open(op_list_path, 'w') as f:
        json.dump(op_list, f)


def single_main(route_name=None):
    set_seed()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    # 设置地图名
    if route_name is not None:
        yaml_config['paths']['route_name'] = route_name

    # 初始化DataLoader，传入base_dir
    config = Config(yaml_config, base_dir=base_dir)
    config.out_path = out_path

    config.load_from_yaml()

    solver = MipSolver(config)
    solver.init_model()
    solver.solve_model()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list
    return map_df, op_list


if __name__ == "__main__":
    map_df, op_list = single_main()
    store_results(out_path, map_df, op_list)
