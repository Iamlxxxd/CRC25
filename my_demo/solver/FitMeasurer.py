# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 15:53
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from DESolver import DESolver
from Individual import Individual
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list

# 极大值
CALC_INF = int(float('inf') / 2)


class FitMeasurer:
    solver: DESolver
    router: Router

    def __init__(self, solver: DESolver):
        self.solver = solver
        self.router = Router(heuristic=solver.heuristic, CRS=solver.meta_map["CRS"], CRS_map=solver.meta_map["CRS_map"])

    def do_measure(self, individual: Individual) -> float:
        null_solution_flag = self.calc_gen_obj(individual)
        if not null_solution_flag:
            return CALC_INF
        cost = self.obj_to_cost(individual)
        return cost

    def calc_gen_obj(self, individual: Individual):
        try:
            path_list, path_graph, path_df = self.router.get_route(individual.weight_df,
                                                                   self.solver.origin_node,
                                                                   self.solver.dest_node,
                                                                   self.solver.heuristic_f)

            route_error = 1 - common_edges_similarity_route_df_weighted(path_df,
                                                                        self.solver.df_path_foil,
                                                                        self.solver.attrs_variable_names)

            sub_op_list = get_virtual_op_list(individual.org_df, individual.weight_df, self.attrs_variable_names)

            individual.route_error = route_error
            individual.graph_error = len([op for op in sub_op_list if op[3] == "success"])
            return True
        except Exception as e:
            return False

    def obj_to_cost(self, individual: Individual) -> float:
        # Route error
        route_error = individual.route_error
        # Graph error
        graph_error = individual.graph_error

        # Calculate the cost
        cost = graph_error + self.solver.lagrangian_lambda * route_error
        individual.obj = cost
        return cost
