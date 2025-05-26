# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/23 17:09
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from my_demo.solver.FitMeasurer import CALC_INF
from my_demo.solver.Individual import Individual
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
import random
import numpy as np
from utils.graph_op import graphOperator


class InitFromRandom:

    init_strategy_list: list

    def __init__(self, solver):
        self.solver = solver

    def do_init_pop(self, left, right):

        for i in range(left, right):
            individual = self.generate_neighbor_p()
            self.solver.pop[i] = individual

    def generate_neighbor_p(self):
        """
        refactor:from utils.mthread.generate_neighbor_p
        """
        df_G = self.solver.org_map_df
        graph_operator = graphOperator()

        n_perturbation = 50  # temp
        while True:
            df_perturbed = df_G.copy()

            operator_names = graph_operator.operator_names
            op_list = []
            for i in range(n_perturbation):
                # randomly choose an attribute to perturb
                edge = df_perturbed.sample(1)
                edge_index = edge.index[0]
                # Select a random operator
                operator_name = np.random.choice(operator_names, p=[0.15, 0.15, 0.15, 0.15, 0.4])
                operator = graph_operator.operator_dict[operator_name]
                # Apply the operator with appropriate parameters
                if operator == graph_operator.modify_path_type:
                    # For path_type, we need to provide an option
                    edge_attr = "path_type"
                    options = graph_operator.get_categorical_options(edge_attr)
                    step = np.random.choice(options)
                    result = operator(edge, df_perturbed, step)
                else:
                    # pass the edge, df_modified, step
                    attr_name, bound = graph_operator.get_numerical_bound_op(operator_name)
                    edge_attr = edge[attr_name].iloc[0]
                    step = np.random.choice([bound[1] - edge_attr, bound[0] - edge_attr])
                    result = operator(edge, df_perturbed, step)
                op_list.append((operator_name, (edge_index, edge["geometry"]), step, result))

            individual = Individual(df_perturbed, self.solver.config.user_model)
            # individual.create_network_graph()

            cost = self.solver.fit_measurer.do_measure(individual)
            if cost >= CALC_INF:
                continue
            else:
                return individual
