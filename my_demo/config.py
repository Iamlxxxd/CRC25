import json
import os

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from utils.common_utils import ensure_crs


class Config:
    def __init__(self, config, base_dir=None):
        self.config = config
        self.paths = config['paths']
        self.base_dir = base_dir
        self.route_name = self.paths['route_name']
        self.show_sub_problem = config['params'].get('show_sub_problem', False)
        self.load_all()

    def _full_path(self, *paths):
        # 拼接路径，支持多段
        path = os.path.join(*paths)
        if os.path.isabs(path) or self.base_dir is None:
            return path
        return os.path.join(self.base_dir, path)

    def load_all(self):
        self.load_network_data()
        self.load_foil_data()
        self.load_params()

    def load_network_data(self):
        # 读取metadata.json
        meta_data_path = self._full_path("data/train/routes", self.route_name, "metadata.json")
        with open(meta_data_path, 'r') as f:
            self.meta_data = json.load(f)
        self.meta_map = self.meta_data["map"]
        self.user_model = self.meta_data["user_model"]
        self.attrs_variable_names = self.user_model["attrs_variable_names"]
        self.route_error_delta = self.user_model["route_error_threshold"]

        # 地图文件名
        map_filename = self.meta_map["map_name"]
        self.basic_network_path = self._full_path("data/train/maps", map_filename)
        self.basic_network: GeoDataFrame = gpd.read_file(self.basic_network_path)

        # 起终点坐标csv路径
        self.gdf_coords_path = self._full_path("data/train/routes", self.route_name, "route_start_end.csv")
        self.gdf_coords_loaded: pd.DataFrame = pd.read_csv(self.gdf_coords_path, sep=';')
        self.gdf_coords_loaded['geometry'] = self.gdf_coords_loaded['geometry'].apply(wkt.loads)
        self.gdf_coords_loaded: GeoDataFrame = gpd.GeoDataFrame(self.gdf_coords_loaded, geometry='geometry')

    def load_foil_data(self):
        # foil路径节点json文件路径
        self.foil_json_path = self._full_path("data/train/routes", self.route_name, "foil_route.json")
        with open(self.foil_json_path, 'r') as f:
            self.path_foil = json.load(f)

        # foil路径文件路径
        self.df_path_foil_path = self._full_path("data/train/routes", self.route_name, "foil_route.gpkg")
        self.df_path_foil: GeoDataFrame = gpd.read_file(self.df_path_foil_path)
        self.df_path_foil = ensure_crs(self.df_path_foil, self.meta_map["CRS"])

    def load_params(self):
        self.n_perturbation = self.config['params']['n_perturbation']
        self.operator_p = self.config['params']['operator_p']
        self.heuristic = self.config['params']['heuristic']
        self.heuristic_f = self.config['params']['heuristic_f']
        self.multi_job = self.config['params']['multi_job']
        self.gen_num = self.config['params']['gen_num']
        self.lagrangian_lambda = self.config['params']['lagrangian_lambda']
        self.time_limit = self.config['params']['time_limit']
        self.store_path = self._full_path(self.config['paths'].get('store_path', './outputs/'))
