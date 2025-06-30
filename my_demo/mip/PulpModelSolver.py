# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/16 17:32
@project:    CRC25
"""
from my_demo.config import Config
from my_demo.mip.ModelSolver import ModelSolver
import os
# 禁用Gurobi的网络许可检查
os.environ["GUROBI_RUNLICCHECK"] = "0"  # 关键变量，阻止许可证检查
os.environ["GRB_COMPUTESERVER"] = ""    # 清空服务器地址
import pulp
from collections import defaultdict
import os


class PulpModelSolver(ModelSolver):
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
            self.w[i] = pulp.LpVariable(f"w_[{i}]", lowBound=0, cat=pulp.LpContinuous)

        # Shortest Path Constraints
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model += self.w[j] - self.w[i] <= row_data['c'] + row_data['d'] * self.x[i][j] + big_m * (
                    1 - self.y[i][j]), "sp_opt_[{},{}]".format(i, j)

        for i, j in self.data_holder.foil_route_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model += self.w[j] - self.w[i] >= row_data['c'] + row_data['d'] * self.x[i][j] - big_m * (
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

        # self.model += self.w[self.data_holder.start_node] == 0, "w_start"
        # self.model += self.w[self.data_holder.end_node] == self.data_holder.foil_cost, "w_end"

        # can not modify path_type
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)
            if not row_data['modify_able_path_type']:
                self.model += self.x[i][j] == 0, "no_x_[{},{}]".format(i, j)

        obj = 0
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)
            obj += row_data['Deleta_p'] * self.y[i][j] + row_data['Deleta_n'] * (1 - self.y[i][j]) + self.x[i][j]
        self.model += obj, "obj"

    def solve_model(self, time_limit=3600, gap=0):
        # solver = pulp.GUROBI_CMD(gapRel=gap, timeLimit=time_limit, keepFiles=False,
        #                          logPath=os.path.join(self.config.base_dir, "my_demo", "output", "solver_log.txt"))
        solver = pulp.HiGHS_CMD(gapRel=gap, timeLimit=time_limit, keepFiles=False,
                                  logPath=os.path.join(self.config.base_dir, "my_demo", "output", "solver_log.txt"))
        self.model.solve(solver)
