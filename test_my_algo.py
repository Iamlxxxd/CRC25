# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/7/9 17:53
@project:    CRC25
"""

import os
import sys
from unittest import TestCase

import pandas as pd

# 获取当前文件所在目录（即submission目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（即submission的父目录）
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # 将项目根目录加入路径

from all_my_algo_gate import *

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

base_dir = os.path.abspath(current_dir)
out_path = os.path.join(base_dir, "my_demo", "output", "submission_out")
experiments_path = os.path.join(base_dir, "my_demo", "output", "experiments")

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


def run_single(route_name):
    print(f"Testing route: {route_name}")
    result = single_hybrid(route_name=route_name)

    experiments_dict = result.get('experiments_dict', dict())
    experiments_dict['route_name'] = route_name
    print(f"Result for {route_name}: {result}")
    return experiments_dict


class Test(TestCase):
    def test_mip(self):
        return_result = single_mip()

    def test_search(self):
        return_result = single_search()

    def test_search_batch(self):
        routes_dir = os.path.join(base_dir, "data/train/routes")
        route_names = [name for name in os.listdir(routes_dir) if os.path.isdir(os.path.join(routes_dir, name))]

        import multiprocessing

        with multiprocessing.Pool(10) as pool:
            final_result = pool.map(run_single, route_names)

        final_result_df = pd.DataFrame(final_result)
        tag = "mip_cost"

        excel_path = os.path.join(experiments_path, "test.xlsx")

        # 确保 experiments_path 目录存在
        os.makedirs(experiments_path, exist_ok=True)

        # 检查文件是否为合法excel，否则删除重建
        from openpyxl import load_workbook
        def is_valid_excel(path):
            try:
                load_workbook(path)
                return True
            except Exception:
                return False

        if os.path.exists(excel_path) and not is_valid_excel(excel_path):
            os.remove(excel_path)

        # 写入excel的指定sheet，保留其他sheet
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                # 删除已存在的sheet
                try:
                    writer.book.remove(writer.book[tag])
                except Exception:
                    pass
                final_result_df.to_excel(writer, sheet_name=tag, index=False)
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                final_result_df.to_excel(writer, sheet_name=tag, index=False)

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
