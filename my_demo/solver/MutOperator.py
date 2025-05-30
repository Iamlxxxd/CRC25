# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 18:33
@project:    CRC25
"""

import random
import multiprocessing

from my_demo.solver.Individual import Individual
from utils.common_utils import set_seed
import numpy as np
import os
import time

class MutOperator:

    def __init__(self, solver):
        self.solver = solver

    def do_move(self):
        self.edge_mutation()

    def edge_mutation(self, row_num: int = None):
        """
        Args:
            row_num:随机抽取指定数量的边进行计算
        Returns:
        """

        pop = self.solver.pop

        args = []
        for i in range(self.solver.pop_size):
            idxs = np.random.choice(len(pop), 3, replace=False)
            r1, r2, r3 = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
            args.append((
                i, row_num, (r1, r2, r3),
                self.solver.operate_columns,
                self.solver.map_constraint,
                self.solver.F,
                self.solver.config.user_model,
                self.solver.fit_measurer,
                self.solver.CALC_INF
            ))

        if self.solver.config.multi_job:
            process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)

            with process_pool as pool:
                mut_pop = pool.starmap(self.mutate_one, args)
            self.solver.mut_pop = mut_pop
        else:
            for i in range(self.solver.pop_size):
                self.solver.mut_pop[i] = self.mutate_one(*args[i])

    def mutate_one(
        self, i, row_num, triple,
        operate_columns, map_constraint, F, user_model, fit_measurer, CALC_INF
    ):
        np.random.seed(os.getpid() + int(time.time()))

        r1, r2, r3 = triple
        new_df = r1.weight_df.copy()
        if row_num is not None and row_num < len(new_df):
            chosen_idx = np.random.choice(list(new_df.index), row_num, replace=False)
        else:
            chosen_idx = list(new_df.index)

        for col in operate_columns:
            v1 = r1.weight_df[col]
            v2 = r2.weight_df[col]
            v3 = r3.weight_df[col]
            new_col = v1.copy()
            if col == "path_type":
                options = map_constraint[col]["categorical_options"]
                mask = (v2 != v3)
                chosen_mask = mask.loc[chosen_idx]
                for idx in chosen_mask.index:
                    if chosen_mask.loc[idx]:
                        if np.random.rand() < 0.5:
                            cur = v1.loc[idx]
                            new_col.loc[idx] = [o for o in options if o != cur][0]
                        else:
                            new_col.loc[idx] = v1.loc[idx]
                    else:
                        new_col.loc[idx] = v1.loc[idx]
                new_df[col] = new_col
            else:
                calc_col = v1 + F * (v2 - v3)
                lb, ub = map_constraint[col]["bound"]
                calc_col = calc_col.where((calc_col <= ub) & (calc_col >= lb), v1)
                new_col.loc[chosen_idx] = calc_col.loc[chosen_idx]
                new_df[col] = new_col

        new_ind = Individual(new_df, user_model)
        cost = fit_measurer.do_measure(new_ind)
        if cost >= CALC_INF:
            return r1.weight_df.copy()
        else:
            return new_ind

