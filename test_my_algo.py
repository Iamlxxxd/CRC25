# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/7/9 17:53
@project:    CRC25
"""
import os
import sys

import os
import sys
import yaml
from unittest import TestCase
from visual import visual_map_foil_modded

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
from config import Config
from src.solver.SearchSolver import SearchSolver
from src.calc.common_utils import set_seed

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

base_dir = os.path.abspath(current_dir)
out_path = os.path.join(base_dir, "my_demo", "output", "submission_out")

mip_visual_path = os.path.join(out_path, "mip")
search_visual_path = os.path.join(out_path, "search")
hybrid_visual_path = os.path.join(out_path, "hybrid")

VISUAL = True


class Test(TestCase):
    def test_mip(self):
        map_df, op_list = self.single_mip()

    def test_search(self):
        map_df, op_list = self.single_search()

    def test_hybrid(self):
        map_df, op_list = self.single_hybrid()

    def single_mip(self, route_name=None):
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
        solver.init_from_config()

        solver.init_model()
        solver.solve_model()

        solver.process_solution_from_model()
        map_df = solver.out_put_df
        op_list = solver.out_put_op_list

        if VISUAL:
            os.makedirs(mip_visual_path, exist_ok=True)
            visual_map_foil_modded(solver.process_visual_data(), mip_visual_path, config.route_name)

        return map_df, op_list

    def single_search(self, route_name=None):
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

        solver = SearchSolver(config)
        solver.init_from_config()

        solver.do_solve()

        solver.process_solution_from_model()
        map_df = solver.out_put_df
        op_list = solver.out_put_op_list

        if VISUAL:
            os.makedirs(search_visual_path, exist_ok=True)
            visual_map_foil_modded(solver.process_visual_data(), search_visual_path, config.route_name)

        return map_df, op_list

    def single_hybrid(self, route_name=None):
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

        mip_solver = MipSolver(config)
        mip_solver.init_from_config()

        mip_solver.init_model()
        mip_solver.solve_model()

        mip_solver.process_solution_from_model()
        mip_solver.modify_arc_solution
        if VISUAL:
            os.makedirs(hybrid_visual_path, exist_ok=True)
            visual_map_foil_modded(mip_solver.process_visual_data(), hybrid_visual_path, config.route_name + "_mip")

        if mip_solver.route_error <= 0 and mip_solver.graph_error <= 1:
            # 改一条边或者不改 已经是最优了  不需要继续搜了 搜也不可能搜到更好的

            return mip_solver.out_put_df, mip_solver.out_put_op_list

        search_solver = SearchSolver(config)
        search_solver.init_from_other_solver(mip_solver)
        search_solver.do_solve()

        search_solver.process_solution_from_model()
        if VISUAL:
            os.makedirs(hybrid_visual_path, exist_ok=True)
            visual_map_foil_modded(search_solver.process_visual_data(), hybrid_visual_path,
                                   config.route_name)

        return search_solver.out_put_df, search_solver.out_put_op_list
