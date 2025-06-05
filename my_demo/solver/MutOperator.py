import random
import multiprocessing
from copy import deepcopy
import os
import time

from my_demo.solver.Individual import Individual


class MutOperator:

    def __init__(self, solver):
        self.solver = solver

    def do_move(self):
        self.edge_mutation()

    def edge_mutation(self, row_num: int = None):
        pop = self.solver.pop
        F = self.solver.F
        operate_columns = self.solver.operate_columns
        map_constraint = self.solver.map_constraint
        user_model = self.solver.config.user_model
        fit_measurer = self.solver.fit_measurer
        CALC_INF = self.solver.CALC_INF
        pop_size = self.solver.pop_size

        mut_pop = [None] * pop_size
        args = []

        global_seed = int(time.time())  # 用于种子生成
        for i in range(pop_size):
            r1, r2, r3 = random.sample(pop, 3)
            rng = random.Random(global_seed + i)  # 每个个体使用独立种子
            args.append((
                r1, r2, r3, row_num,
                operate_columns, map_constraint, F,
                user_model, fit_measurer, CALC_INF,
                rng
            ))

        if self.solver.config.multi_job:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
                mut_pop = pool.starmap(MutOperator._mutate_one_static, args)
        else:
            for i in range(pop_size):
                mut_pop[i] = self._mutate_logic(*args[i])

        self.solver.mut_pop = mut_pop

    @staticmethod
    def _mutate_one_static(*args):
        return MutOperator._mutate_logic(*args)

    @staticmethod
    def _mutate_logic(
        r1, r2, r3, row_num,
        operate_columns, map_constraint, F,
        user_model, fit_measurer, CALC_INF,
        rng  # random.Random 实例
    ):
        new_df = r1.weight_df.copy()
        if row_num is not None and row_num < len(new_df):
            chosen_idx = rng.sample(list(new_df.index), row_num)
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
                        cur = v1.loc[idx]
                        new_col.loc[idx] = [o for o in options if o != cur][0] if rng.random() < 0.5 else v1.loc[idx]
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

        return deepcopy(r1) if cost >= CALC_INF else new_ind