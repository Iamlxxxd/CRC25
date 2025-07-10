# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/7/9 17:53
@project:    CRC25
"""

import os
import sys
import yaml
from unittest import TestCase
import time

from visual import visual_map_foil_modded

# 获取当前文件所在目录（即submission目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（即submission的父目录）
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # 将项目根目录加入路径

from src.solver.MipSolver import MipSolver
from config import Config
from src.solver.SearchSolver import SearchSolver
from src.utils.common_utils import set_seed
from all_my_algo_gate import *

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

base_dir = os.path.abspath(current_dir)
out_path = os.path.join(base_dir, "my_demo", "output", "submission_out")

mip_visual_path = os.path.join(out_path, "mip")
search_visual_path = os.path.join(out_path, "search")
hybrid_visual_path = os.path.join(out_path, "hybrid")

ROUTE_NAME = "osdpm_0_3"

test_args = {
    "meta_data_path": os.path.join(base_dir, "data/train/routes", ROUTE_NAME, "metadata.json"),
    "basic_network_path": os.path.join(base_dir, "data/train/maps/osdpm_map.gpkg"),  # 这个如果地图名字换了需要看metadata.json里写的地图
    "foil_json_path": os.path.join(base_dir, "data/train/routes", ROUTE_NAME, "foil_route.json"),
    "df_path_foil_path": os.path.join(base_dir, "data/train/routes", ROUTE_NAME, "foil_route.gpkg"),
    "gdf_coords_path": os.path.join(base_dir, "data/train/routes", ROUTE_NAME, "route_start_end.csv"),
    "output_path": os.path.join(base_dir, "data/train/routes", ROUTE_NAME, "submission_out")
}
import argparse

my_args = argparse.Namespace(**test_args)

class Test(TestCase):
    def test_mip(self):
        return_result = single_mip()

    def test_search(self):
        return_result = single_search()


    def test_hybrid(self):
        return_result = single_hybrid()

    def test_mip_from_args(self):
        return_result = single_mip_from_args(my_args)

    def test_search_from_args(self):

        return_result = single_search_from_args(my_args)

    def test_hybrid_from_args(self):
        return_result = single_hybrid_from_args(my_args)

    def test_multi_job(self):
        return_result = multi_job(my_args)
