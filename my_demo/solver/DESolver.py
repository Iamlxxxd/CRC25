# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 16:02
@project:    CRC25
"""
from networkx.classes import DiGraph

from my_demo.solver.Individual import Individual
from typing import List
from geopandas import GeoDataFrame

from my_demo.config import Config
from my_demo.solver.CrossOperator import CrossOperator
from my_demo.solver.FitMeasurer import FitMeasurer
from my_demo.solver.MutOperator import MutOperator
from my_demo.solver.PopInit.PopInitializer import PopInitializer
from my_demo.solver.SelectOperator import SelectOperator
from router import Router
from geopandas import GeoDataFrame
from jupyter_server.auth import User
from networkx.classes import MultiDiGraph

from utils.dataparser import create_network_graph, handle_weight, handle_weight_with_recovery
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from copy import deepcopy


class DESolver:
    org_map_df: GeoDataFrame
    org_graph: DiGraph

    origin_node: tuple
    dest_node: tuple

    meta_map: dict
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

    pop_size = 50
    pop: List[Individual] = [None] * pop_size
    mut_pop: List[Individual] = [None] * pop_size
    cross_pop: List[Individual] = [None] * pop_size

    map_constraint: dict = {
        "obstacle_free_width_float": {"bound": [0.6, 2]},
        "curb_height_max": {"bound": [0, 0.2]},
        "path_type": {"categorical_options": ["walk", "bike"], "bound": [0, 1]},
    }

    operate_columns: list = [
        "obstacle_free_width_float",
        "curb_height_max",
        "path_type"
    ]

    F: float = 0.5
    PROB_CROSS: float = 0.001,
    MAX_ITER: int = 2000
    lagrangian_lambda: int = 2000

    router: Router

    def __init__(self, config: Config):
        self.config = config
        self.meta_map = config.meta_map

        self.router = Router(heuristic=self.heuristic, CRS=self.meta_map["CRS"], CRS_map=self.meta_map["CRS_map"])
        self.load_basic_data()

        self.initializer = PopInitializer(self)
        self.fit_measurer = FitMeasurer(self)
        self.mut_operator = MutOperator(self)
        self.cross_operator = CrossOperator(self)
        self.select_operator = SelectOperator(self)

    def load_basic_data(self):
        self.org_map_df = self.config.basic_network
        df_copy = deepcopy(self.org_map_df)
        df_copy = handle_weight(df_copy, self.config.user_model)
        _, self.org_graph = create_network_graph(df_copy)

        self.origin_node, self.dest_node, _, _, _ = self.router.set_o_d_coords(self.org_graph,
                                                                               self.config.gdf_coords_loaded)

    def run(self, max_iter=None):
        self.initializer.heuristic_init_pop()

        self.MAX_ITER = max_iter or self.MAX_ITER
        for i in range(self.MAX_ITER):
            self.mut_operator.do_move()
            self.cross_operator.do_move()
            self.select_operator.do_move()
