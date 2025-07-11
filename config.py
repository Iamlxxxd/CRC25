import os
import json
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt

from src.utils.common_utils import ensure_crs

class Config:
    def __init__(self, config=None, base_dir=None):
        self.config = config
        self.paths = config['paths'] if config else {}
        self.base_dir = base_dir
        self.route_name = self.paths.get('route_name', '') if config else ''

    def _full_path(self, *paths):
        # 拼接路径，支持多段
        path = os.path.join(*paths)
        if os.path.isabs(path) or self.base_dir is None:
            return path
        return os.path.join(self.base_dir, path)

    def load_from_yaml(self):
        self.load_network_data()
        self.load_foil_data()
        self.load_params()
    
    def load_from_args(self, args):
        # 从args中加载网络数据
        self._load_network_data_from_args(args)
        # 从args中加载foil数据
        self._load_foil_data_from_args(args)
        # 设置默认参数
        self.load_params()

    def _load_network_data_from_args(self, args):
        # 读取metadata.json
        with open(args.meta_data_path, 'r') as f:
            self.meta_data = json.load(f)
        self.meta_map = self.meta_data["map"]
        self.user_model = self.meta_data["user_model"]
        self.attrs_variable_names = self.user_model["attrs_variable_names"]
        self.route_error_delta = self.user_model["route_error_threshold"]

        # 基础网络路径
        self.basic_network_path = args.basic_network_path
        self.basic_network: GeoDataFrame = gpd.read_file(self.basic_network_path)

        # 起终点坐标csv路径
        self.gdf_coords_path = args.gdf_coords_path
        self.gdf_coords_loaded: pd.DataFrame = pd.read_csv(self.gdf_coords_path, sep=';')
        self.gdf_coords_loaded['geometry'] = self.gdf_coords_loaded['geometry'].apply(wkt.loads)
        self.gdf_coords_loaded: GeoDataFrame = gpd.GeoDataFrame(self.gdf_coords_loaded, geometry='geometry')

    def _load_foil_data_from_args(self, args):
        # foil路径节点json文件路径
        self.foil_json_path = args.foil_json_path
        with open(self.foil_json_path, 'r') as f:
            self.path_foil = json.load(f)

        # foil路径文件路径
        self.df_path_foil_path = args.df_path_foil_path
        self.df_path_foil: GeoDataFrame = gpd.read_file(self.df_path_foil_path)
        self.df_path_foil = ensure_crs(self.df_path_foil, self.meta_map["CRS"])

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
        self.heuristic = self.config['params']['heuristic']
        self.heuristic_f = self.config['params']['heuristic_f']
        self.store_path = self._full_path(self.config['paths'].get('store_path', './outputs/'))
        self.visual = self.config['params']['visual']
