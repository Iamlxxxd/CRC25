# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 18:33
@project:    CRC25
"""
from geopandas import GeoDataFrame
from networkx.classes import DiGraph

from DESolver import DESolver
from Individual import Individual
from router import Router
from typing import List, Tuple
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
import random


class MutOperator:
    solver: DESolver

    def __init__(self, solver: DESolver):
        self.solver = solver

    def do_move(self):
        self.edge_mutation()

    def edge_mutation(self, row_num: int = None):
        pop = self.solver.pop
        mut_pop = []
        F = self.solver.F
        operate_columns = self.solver.operate_columns
        map_constraint = self.solver.map_constraint

        for i in range(self.solver.pop_size):
            triple = random.sample(pop, 3)
            r1, r2, r3 = triple[0], triple[1], triple[2]
            new_df = r1.route_df.copy()
            # 随机抽取row_num个索引
            if row_num is not None and row_num < len(new_df):
                chosen_idx = random.sample(list(new_df.index), row_num)
            else:
                chosen_idx = list(new_df.index)

            for col in operate_columns:
                v1 = r1.route_df[col]
                v2 = r2.route_df[col]
                v3 = r3.route_df[col]
                new_col = v1.copy()
                if col == "path_type":
                    options = map_constraint[col]["categorical_options"]
                    mask = (v2 != v3)
                    # 只对chosen_idx批量操作
                    chosen_mask = mask.loc[chosen_idx]
                    for idx in chosen_mask.index:
                        if chosen_mask.loc[idx]:
                            if random.random() < 0.5:
                                cur = v1.loc[idx]
                                new_col.loc[idx] = [o for o in options if o != cur][0]
                            else:
                                new_col.loc[idx] = v1.loc[idx]
                        else:
                            new_col.loc[idx] = v1.loc[idx]
                    # 其他未选中的直接赋值（其实new_col本来就是v1的copy，可以省略）
                    new_df[col] = new_col
                else:
                    calc_col = v1 + F * (v2 - v3)
                    lb, ub = map_constraint[col]["bound"]
                    calc_col = calc_col.where((calc_col <= ub) & (calc_col >= lb), v1)
                    # 只对chosen_idx批量赋值
                    new_col.loc[chosen_idx] = calc_col.loc[chosen_idx]
                    new_df[col] = new_col

            new_ind = Individual(new_df, self.solver.user_model)
            new_ind.create_network_graph()

            self.solver.fit_measurer.do_measure(new_ind)
            mut_pop.append(new_ind)

        self.solver.mut_pop = mut_pop
