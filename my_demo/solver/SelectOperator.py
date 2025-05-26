# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/26 14:39
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from my_demo.solver.Individual import Individual
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
import random


class SelectOperator:

    def __init__(self, solver):
        self.solver = solver

    def do_move(self):
        new_pop = []
        for ind1, ind2 in zip(self.solver.pop, self.solver.cross_pop):
            if ind1.obj <= ind2.obj:
                new_pop.append(ind1)
            else:
                new_pop.append(ind2)
        self.solver.pop = new_pop
