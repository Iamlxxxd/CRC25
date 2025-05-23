# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 17:00
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from DESolver import DESolver
from Individual import Individual
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list

class CrossOperator:
    solver: DESolver

    def __init__(self, solver: DESolver):
        self.solver = solver