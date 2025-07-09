import os
import sys

# 在所有入口文件（如 submission_template.py）顶部添加：
import os
import sys

# 获取当前文件所在目录（即submission目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（即submission的父目录）
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # 将项目根目录加入路径

import argparse
import json

from config import Config
from src.solver.SearchSolverSaturated import SearchSolverSaturated
from utils.dataparser import convert_op_list_to_wkt

def get_results(args):
    # TODO: Implement this function with your own algorithm
    map_df = None
    op_list = None
    config = Config()
    config.load_from_args(args)

    solver = SearchSolverSaturated(config)
    solver.do_solve()

    solver.process_solution_from_model()
    map_df = solver.out_put_df
    op_list = solver.out_put_op_list
    return map_df, op_list


def store_results(output_path, map_df, op_list):
    map_df_path = os.path.join(output_path, "map_df.gpkg")
    op_list_path = os.path.join(output_path, "op_list.json")

    map_df.to_file(map_df_path, driver='GPKG')
    with open(op_list_path, 'w') as f:
        json.dump(op_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_data_path", type=str, required=True)
    parser.add_argument("--basic_network_path", type=str, required=True)
    parser.add_argument("--foil_json_path", type=str, required=True)
    parser.add_argument("--df_path_foil_path", type=str, required=True)
    parser.add_argument("--gdf_coords_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    map_df, op_list = get_results(args)
    store_results(args.output_path, map_df, op_list)
