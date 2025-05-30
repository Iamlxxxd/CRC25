import json
import os

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt


class Config:
    def __init__(self, config, base_dir=None):
        self.config = config
        self.paths = config['paths']
        # 基础路径（如 examples 文件夹），用于拼接所有相对路径
        self.base_dir = base_dir
        self.load_all()

    def _full_path(self, path):
        # 如果是绝对路径，直接返回，否则拼接base_dir
        if os.path.isabs(path) or self.base_dir is None:
            return path
        return os.path.join(self.base_dir, path)

    def load_all(self):
        self.load_network_data()
        self.load_foil_data()
        self.load_params()

    def load_network_data(self):
        # 基础网络文件路径
        self.basic_network_path = self._full_path(self.paths['basic_network_path'])
        # 基础网络数据（GeoDataFrame）
        self.basic_network: GeoDataFrame = gpd.read_file(self.basic_network_path)

        # 起终点坐标csv路径
        self.gdf_coords_path = self._full_path(self.paths['gdf_coords_path'])
        # 起终点坐标（DataFrame）
        self.gdf_coords_loaded: pd.DataFrame = pd.read_csv(self.gdf_coords_path, sep=';')
        self.gdf_coords_loaded['geometry'] = self.gdf_coords_loaded['geometry'].apply(wkt.loads)
        # 起终点坐标（GeoDataFrame）
        self.gdf_coords_loaded: GeoDataFrame = gpd.GeoDataFrame(self.gdf_coords_loaded, geometry='geometry')

    def load_foil_data(self):
        # foil路径节点json文件路径
        self.foil_json_path = self._full_path(self.paths['foil_json_path'])
        # foil路径节点数据（dict）
        with open(self.foil_json_path, 'r') as f:
            self.path_foil = json.load(f)

        # foil路径文件路径
        self.df_path_foil_path = self._full_path(self.paths['df_path_foil_path'])
        # foil路径数据（GeoDataFrame）
        self.df_path_foil: GeoDataFrame = gpd.read_file(self.df_path_foil_path)

        # 元数据json路径
        self.meta_data_path = self._full_path(self.paths['meta_data_path'])
        # 元数据（dict）
        with open(self.meta_data_path, 'r') as f:
            self.meta_data = json.load(f)

        # 用户模型参数（dict）
        self.user_model = self.meta_data["user_model"]
        # 地图相关元数据（dict）
        self.meta_map = self.meta_data["map"]

        # 可变属性名列表
        self.attrs_variable_names = self.user_model["attrs_variable_names"]
        # 路径误差阈值
        self.route_error_delta = self.user_model["route_error_threshold"]

    def load_params(self):
        # 扰动次数
        self.n_perturbation = self.config['params']['n_perturbation']
        # 操作概率分布
        self.operator_p = self.config['params']['operator_p']
        # 路径搜索启发式算法
        self.heuristic = self.config['params']['heuristic']
        # 路径权重字段
        self.heuristic_f = self.config['params']['heuristic_f']
        # 是否并行
        self.multi_job = self.config['params']['multi_job']
        # 迭代代数
        self.gen_num = self.config['params']['gen_num']
        # 拉格朗日惩罚系数
        self.lagrangian_lambda = self.config['params']['lagrangian_lambda']
        # 时间限制（秒）
        self.time_limit = self.config['params']['time_limit']
        # 结果存储路径
        self.store_path = self._full_path(self.config['paths'].get('store_path', './outputs/'))
