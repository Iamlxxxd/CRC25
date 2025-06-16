# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/16 17:32
@project:    CRC25
"""
from my_demo.config import Config
from my_demo.mip.MipModelSolver import ModelSolver
import pulp
from collections import defaultdict


class ModelSolverNew(ModelSolver):
    def __init__(self, config: Config):
        super().__init__(config)

    def init_model(self):

        self.model = pulp.LpProblem("CRC25", pulp.LpMinimize)
        big_m = self.data_holder.M * 4
        # 定义决策变量
        self.x = defaultdict(dict)
        self.y = defaultdict(dict)
        self.w = dict()
        for i, j in self.data_holder.all_arcs:
            self.x[i].update({j: pulp.LpVariable(f"x_[{i},{j}]", cat=pulp.LpBinary)})

            self.y[i].update({j: pulp.LpVariable(f"y_[{i},{j}]", cat=pulp.LpBinary)})

        for i in self.data_holder.all_nodes:
            self.w[i] = pulp.LpVariable(f"w_[{i}]", cat=pulp.LpContinuous)

        # Shortest Path Constraints
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model += self.w[j] - self.w[i] <= row_data['c'] + row_data['d'] * self.x[i][j] + big_m * (
                    1 - self.y[i][j]), "sp_opt_[{},{}]".format(i, j)

        for i, j in self.data_holder.foil_route_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model += self.w[j] - self.w[i] <= row_data['c'] + row_data['d'] * self.x[i][j] - big_m * (
                    1 - self.y[i][j]), "sp_foil_[{},{}]".format(i, j)
            self.model += self.y[i][j] == 1, "foil_y_[{},{}]".format(i, j)

        # Undirected Arc Constraint
        for f, arcs in self.data_holder.all_feasible_both_way.items():
            for i, j in arcs:
                # todo 实际上不需要f 数据没处理
                self.model += self.x[i][j] == self.x[j][i], "ua_pos_[{},{},{}]".format(f, i, j)
                self.model += self.y[i][j] == self.y[j][i], "ua_y_pos_[{},{},{}]".format(f, i, j)

        for f, arcs in self.data_holder.all_infeasible_both_way.items():
            for i, j in arcs:
                self.model += self.x[i][j] == self.x[j][i], "ua_neg_[{},{},{}]".format(f, i, j)
                self.model += self.y[i][j] == self.y[j][i], "ua_y_neg_[{},{},{}]".format(f, i, j)

        self.model += self.w[self.data_holder.start_node] == 0, "w_start"
        self.model += self.w[self.data_holder.end_node] == self.data_holder.foil_cost, "w_end"

        obj = 0
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)
            obj += row_data['Deleta_p'] * self.y[i][j] + row_data['Deleta_n'] * (1 - self.y[i][j]) + self.x[i][j]

        self.model += obj, "obj"

    def solve_model(self, time_limit=3600, gap=0.01):
        solver = pulp.GUROBI_CMD(gapRel=gap, timeLimit=time_limit,keepFiles=False)
        self.model.solve(solver)

    def modify_org_map_df_by_solution(self):
        # todo unfinished
        modify_dict = defaultdict(list)
        self.graph_error = 0
        for (f, i, j), value in self.x_pos.items():
            if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                continue
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(attr_name, i, j, tag="to_infe")

                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (f, i, j), value in self.x_neg.items():
            if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                continue
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(attr_name, i, j, tag="to_fe")

                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (i, j), value in self.x_p.items():
            if ((i, j) in modify_dict) or ((j, i) in modify_dict):
                continue
            if value.X > 0.99:
                self.modify_df_arc_with_attr("path_type", i, j, tag="change")

                self.graph_error += 1
                modify_dict[(i, j)].append("path_type")
