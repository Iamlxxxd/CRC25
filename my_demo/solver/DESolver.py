# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 16:02
@project:    CRC25
"""
from Individual import Individual
from typing import List
from geopandas import GeoDataFrame

from my_demo.solver.CrossOperator import CrossOperator
from my_demo.solver.FitMeasurer import FitMeasurer
from my_demo.solver.MutOperator import MutOperator
from my_demo.solver.PopInit.PopInitializer import PopInitializer


class DESolver:
    org_map_df: GeoDataFrame

    origin_node: tuple
    dest_node: tuple
    path_fuc: str

    meta_map: dict
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

    pop_size: int
    pop: List[Individual]
    mut_pop: List[Individual]
    cross_pop: List[Individual]

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

    def __init__(self):
        self.initializer = PopInitializer(self)
        self.mut_operator = MutOperator(self)
        self.cross_operator = CrossOperator(self)
        self.fit_measurer = FitMeasurer(self)
        # self.mut_operator = MutOperator(self)

    def run(self, max_iter=None):
        self.initializer.heuristicInitPop()

        self.max_iter = max_iter or self.max_iter
        for i in range(self.max_iter):
            self.mut_operator.do_move()
            self.cross_operator.do_move()
            # self.selection() todo
