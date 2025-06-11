# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 16:02
@project:    CRC25
"""
from copy import deepcopy
from typing import List
import multiprocessing

import numpy as np
from geopandas import GeoDataFrame
from networkx.classes import DiGraph
from tqdm import tqdm

from my_demo.config import Config
from my_demo.solver.CrossOperator import CrossOperator
from my_demo.solver.FitMeasurer import FitMeasurer
from my_demo.solver.Individual import Individual
from my_demo.solver.MutOperator import MutOperator
from my_demo.solver.PopInit.PopInitializer import PopInitializer
from my_demo.solver.SelectOperator import SelectOperator
from router import Router
from utils.dataparser import create_network_graph, handle_weight
from utils.common_utils import set_seed


class DESolver:
    org_map_df: GeoDataFrame
    org_graph: DiGraph

    origin_node: tuple
    dest_node: tuple

    df_path_foil: GeoDataFrame
    df_path_fact: GeoDataFrame

    meta_map: dict
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

    # 极大值
    CALC_INF = int(np.finfo(np.float64).max / 2)

    best_individual: Individual = None
    each_iter_best_individual: List[Individual] = []

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
    PROB_CROSS: float = 0.2
    MAX_ITER: int = 2000

    no_improve_count: int = 0
    no_improve_time_rate: float = 0.2

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

        self.no_improve_count = 0

    def load_basic_data(self):
        self.org_map_df = self.config.basic_network
        df_copy = deepcopy(self.org_map_df)
        df_copy = handle_weight(df_copy, self.config.user_model)
        _, self.org_graph = create_network_graph(df_copy)

        self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, _ = self.router.set_o_d_coords(
            self.org_graph,
            self.config.gdf_coords_loaded)
        self.path_fact, self.G_path_fact, self.df_path_fact = self.router.get_route(self.org_graph, self.origin_node,
                                                                                    self.dest_node, self.heuristic_f)
        self.df_path_foil = self.config.df_path_foil

        # 为什么放到这里？ multiprocessing fork的时候 如果不在init初始化 会fork一份内容为空的
        self.pop_size = 50
        self.pop: List[Individual] = [None] * self.pop_size
        self.mut_pop: List[Individual] = [None] * self.pop_size
        self.cross_pop: List[Individual] = [None] * self.pop_size

    def run(self, max_iter=None):
        self.initializer.heuristic_init_pop()

        self.MAX_ITER = max_iter or self.MAX_ITER

        pbar = tqdm(range(self.MAX_ITER))
        for i in pbar:
            self.mut_operator.do_move()
            self.cross_operator.do_move()
            self.select_operator.do_move()
            # rank
            self.pop = sorted(self.pop, key=lambda x: x.obj)

            current_best = self.pop[0]
            flag = "="
            if self.best_individual is None \
                    or current_best.obj < self.best_individual.obj:

                best_cp = deepcopy(current_best)
                self.best_individual = best_cp
                self.each_iter_best_individual.append(best_cp)
                self.no_improve_count = 0
                flag = "↑"
            else:
                self.each_iter_best_individual.append(self.best_individual)
                self.no_improve_count += 1

            pbar.set_postfix({'obj': str(self.best_individual), "update": flag, "stay_count": self.no_improve_count})
            if self.termination_trigger(i):
                break

    def termination_trigger(self, iter_times: int) -> bool:
        if self.no_improve_count >= self.MAX_ITER * self.no_improve_time_rate:
            return True

        return False

    def process_visual_data(self) -> dict:

        return {"gdf_coords":self.config.gdf_coords_loaded,
                "origin_node_loc_length":self.origin_node_loc,
                "dest_node_loc_length":self.dest_node_loc,
                "meta_map":self.meta_map,
                "df_path_fact":self.df_path_fact,
                "df_path_foil":self.df_path_foil,
                "best_route":self.best_individual.path_df,
                "org_map_df":self.org_map_df}

    def process_visual_iter_data(self)-> dict:
        each_iter_best_individual = solver.each_iter_best_individual

        objs = [ind.obj for ind in each_iter_best_individual]
        graph_errors = [ind.graph_error for ind in each_iter_best_individual]
        route_errors = [ind.route_error for ind in each_iter_best_individual]
        return {"objs":objs,
                "graph_errors":graph_errors,
                "route_errors":route_errors}