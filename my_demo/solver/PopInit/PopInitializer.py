# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/23 16:50
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from my_demo.solver.DESolver import DESolver
from my_demo.solver.Individual import Individual
from my_demo.solver.PopInit.InitFromRandom import InitFromRandom
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
import random
import math


class PopInitializer:
    solver: DESolver

    init_strategy_list: list

    def __init__(self, solver: DESolver):
        self.solver = solver
        self.__set_strategy_list()

    def __set_strategy_list(self):
        self.init_strategy_list.append(InitFromRandom(self.solver))
        # todo other

    def heuristicInitPop(self):
        pop_size = self.solver.pop_size
        gradient = int(math.ceil(pop_size / len(self.init_strategy_list)))
        left = 0
        right = gradient

        for each_strategy in self.init_strategy_list:
            if each_strategy == self.init_strategy_list[-1]:
                right = pop_size
            each_strategy.do_init_pop(left, right)
            left = right
            right = right + gradient
