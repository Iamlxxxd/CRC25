import os
import yaml
from my_demo.data_loader import DataLoader
import sys
import os
import random
import numpy as np
sys.path.append("..")


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

    # 自动查找examples目录作为基础路径
    # 假设config.yaml和my_demo在同级，examples与其同级
    base_dir = os.path.join(current_dir, "..", "examples")
    base_dir = os.path.abspath(base_dir)

    # 初始化DataLoader，传入base_dir
    data_loader = DataLoader(config, base_dir=base_dir)

    # 构造args
    args = {
        'basic_network_path': data_loader.basic_network_path,
        'foil_json_path': data_loader.foil_json_path,
        'df_path_foil_path': data_loader.df_path_foil_path,
        'gdf_coords_path': data_loader.gdf_coords_path,
        'heuristic': data_loader.heuristic,
        'heuristic_f': data_loader.heuristic_f,
        'jobs': data_loader.jobs,
        'attrs_variable_names': data_loader.attrs_variable_names,
        "n_perturbation": data_loader.n_perturbation,
        "operator_p": data_loader.operator_p,
        "user_model": data_loader.user_model,
        "meta_map": data_loader.meta_map
    }



if __name__ == "__main__":
    main()

