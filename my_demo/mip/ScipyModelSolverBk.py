#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/29 17:43
# @Author  : JunhaoShi(01387247)
# @Desc    :
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

from my_demo.config import Config
from my_demo.mip.ModelSolver import ModelSolver


class ScipyModelSolver(ModelSolver):
    def __init__(self, config: Config):
        super().__init__(config)
        self.big_m = self.data_holder.M * 4

    def init_model(self):
        # ========== 1. 变量索引映射 ==========
        nodes = sorted(self.data_holder.all_nodes)
        arcs = sorted(self.data_holder.all_arcs)  # 所有有向弧

        # 创建索引映射
        var_index = {}
        idx = 0

        # w_i 变量
        w_index = {node: idx for idx, node in enumerate(nodes)}
        idx += len(nodes)

        # x_ij 变量
        x_index = {}
        for arc in arcs:
            var_index[('x', arc)] = idx
            x_index[arc] = idx
            idx += 1

        # y_ij 变量
        y_index = {}
        for arc in arcs:
            var_index[('y', arc)] = idx
            y_index[arc] = idx
            idx += 1

        total_vars = idx
        self.var_index = var_index
        self.w_index = w_index
        self.x_index = x_index
        self.y_index = y_index

        # ========== 2. 构建目标向量 c ==========
        c = np.zeros(total_vars)

        for arc in arcs:
            row = self.get_row_info_by_arc(*arc)
            # x_ij 的系数
            c[x_index[arc]] = 1
            # y_ij 的系数
            c[y_index[arc]] = row['Deleta_p'] - row['Deleta_n']

        # ========== 3. 边界条件 ==========
        bounds = []
        # w_i >= 0
        bounds.extend([(0, None)] * len(nodes))
        # 0 <= x_ij <= 1
        bounds.extend([(0, 1)] * len(arcs))
        # 0 <= y_ij <= 1
        bounds.extend([(0, 1)] * len(arcs))

        # ========== 4. 约束构建 ==========
        A_ub = []  # 不等式约束矩阵
        b_ub = []  # 不等式右侧向量
        A_eq = []  # 等式约束矩阵
        b_eq = []  # 等式右侧向量

        # (1) 最短路径约束 (所有弧)
        for (i, j) in arcs:
            row = self.get_row_info_by_arc(i, j)
            c_ij = row['c']
            d_ij = row['d']

            # 初始化约束行
            cons = np.zeros(total_vars)
            # w_j - w_i 项
            cons[w_index[j]] = 1
            cons[w_index[i]] = -1
            # -d_ij * x_ij 项
            cons[x_index[(i, j)]] = -d_ij
            # -big_m * y_ij 项
            cons[y_index[(i, j)]] = self.big_m

            A_ub.append(cons)
            b_ub.append(c_ij + self.big_m)

        # (2) Foil路径约束 (特定弧)
        for (i, j) in self.data_holder.foil_route_arcs:
            row = self.get_row_info_by_arc(i, j)
            c_ij = row['c']
            d_ij = row['d']

            # 约束1: y_ij = 1 (等式)
            cons_eq = np.zeros(total_vars)
            cons_eq[y_index[(i, j)]] = 1
            A_eq.append(cons_eq)
            b_eq.append(1)

            # 约束2: 势能不等式
            cons_ub = np.zeros(total_vars)
            # w_i - w_j 项
            cons_ub[w_index[i]] = 1
            cons_ub[w_index[j]] = -1
            # d_ij * x_ij 项
            cons_ub[x_index[(i, j)]] = d_ij
            # big_m * y_ij 项
            cons_ub[y_index[(i, j)]] = self.big_m

            A_ub.append(cons_ub)
            b_ub.append(self.big_m - c_ij)

        # (3) 无向边对称约束
        for group in self.data_holder.all_feasible_both_way.values():
            for (i, j) in group:
                # x_ij = x_ji
                cons_x = np.zeros(total_vars)
                cons_x[x_index[(i, j)]] = 1
                cons_x[x_index[(j, i)]] = -1
                A_eq.append(cons_x)
                b_eq.append(0)

                # y_ij = y_ji
                cons_y = np.zeros(total_vars)
                cons_y[y_index[(i, j)]] = 1
                cons_y[y_index[(j, i)]] = -1
                A_eq.append(cons_y)
                b_eq.append(0)
        for group in self.data_holder.all_infeasible_both_way.values():
            for (i, j) in group:
                # x_ij = x_ji
                cons_x = np.zeros(total_vars)
                cons_x[x_index[(i, j)]] = 1
                cons_x[x_index[(j, i)]] = -1
                A_eq.append(cons_x)
                b_eq.append(0)

                # y_ij = y_ji
                cons_y = np.zeros(total_vars)
                cons_y[y_index[(i, j)]] = 1
                cons_y[y_index[(j, i)]] = -1
                A_eq.append(cons_y)
                b_eq.append(0)

        # (4) 不可修改约束
        for (i, j) in self.data_holder.all_arcs:
            row = self.get_row_info_by_arc(i, j)
            if not row['modify_able_path_type']:
                cons = np.zeros(total_vars)
                cons[x_index[(i, j)]] = 1
                A_eq.append(cons)
                b_eq.append(0)

        # ========== 5. 保存约束矩阵 ==========
        self.A_ub = np.array(A_ub) if A_ub else None
        self.b_ub = np.array(b_ub) if b_ub else None
        self.A_eq = np.array(A_eq) if A_eq else None
        self.b_eq = np.array(b_eq) if b_eq else None
        self.c = c
        self.bounds = bounds

    def solve_model(self, time_limit=3600, gap=0):
        opt = {'disp': True}
        # 注意: scipy 1.10 不支持时间限制和间隙控制
        result = linprog(
            c=self.c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method='highs',  # HiGHS 求解器
            options=opt
        )
        if not result.success:
            raise RuntimeError(f"求解失败: {result.message}")

        # 将解保存到变量字典
        self.solution = result.x
        self._save_solution_to_vars()

    def _save_solution_to_vars(self):
        """将解向量赋值回字典变量"""
        nodes = sorted(self.data_holder.all_nodes)
        arcs = self.data_holder.all_arcs

        # 初始化字典
        self.w = {node: 0.0 for node in nodes}
        self.x = defaultdict(dict)
        self.y = defaultdict(dict)

        # 填充w
        for node in nodes:
            self.w[node] = self.solution[self.w_index[node]]

        # 填充x和y
        for arc in arcs:
            i, j = arc
            x_val = self.solution[self.x_index[arc]]
            y_val = self.solution[self.y_index[arc]]

            self.x[i][j] = x_val
            self.y[i][j] = y_val

            # 确保对称性
            if (j, i) in arcs:
                self.x[j][i] = x_val
                self.y[j][i] = y_val

    def modify_org_map_df_by_solution(self, exclude_arcs=None):
        if exclude_arcs is None:
            exclude_arcs = []
        # todo unfinished
        feature_modify_mark = set()
        self.graph_error = 0
        for i, j_dict in self.y.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
                if (i, j) in feature_modify_mark or (j, i) in feature_modify_mark:
                    continue
                # if value.varValue >0.99999:
                if value == 1:
                    self.modify_df_arc_with_attr(i, j, "to_fe")
                else:
                    self.modify_df_arc_with_attr(i, j, "to_infe")
                feature_modify_mark.add((i, j))

        type_modify_mark = set()
        for i, j_dict in self.x.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
                if (i, j) in type_modify_mark or (j, i) in type_modify_mark:
                    continue
                if value == 1:
                    self.modify_df_arc_with_attr(i, j, "change")
                    type_modify_mark.add((i, j))

    def fill_w_value_for_visual(self):
        self.org_map_df['w_i'] = 0
        self.org_map_df['w_j'] = 0
        self.org_map_df['y_ij'] = 0

        for idx, row in self.org_map_df.iterrows():
            arc = row['arc']
            self.org_map_df.at[row.name, "w_i"] = self.w[arc[0]]
            self.org_map_df.at[row.name, "w_j"] = self.w[arc[1]]
            self.org_map_df.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]

        self.df_path_foil['w_i'] = 0
        self.df_path_foil['w_j'] = 0
        self.df_path_foil['y_ij'] = 0

        cost = 0
        for idx, row in self.df_path_foil.iterrows():
            arc = row['arc']
            self.df_path_foil.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_path_foil.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_path_foil.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["foil_cost"] = round(cost, 4)

        self.df_best_route['w_i'] = 0
        self.df_best_route['w_j'] = 0
        self.df_best_route['y_ij'] = 0
        cost = 0
        for idx, row in self.df_best_route.iterrows():
            arc = row['arc']
            self.df_best_route.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_best_route.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_best_route.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["best_cost"] = round(cost, 4)

        self.df_path_fact['w_i'] = 0
        self.df_path_fact['w_j'] = 0
        self.df_path_fact['y_ij'] = 0
        cost = 0
        for idx, row in self.df_path_fact.iterrows():
            arc = row['arc']
            self.df_path_fact.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_path_fact.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_path_fact.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["fact_cost"] = round(cost, 4)
        pass