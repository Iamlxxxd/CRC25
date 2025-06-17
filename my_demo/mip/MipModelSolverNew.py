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
import os


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
        solver = pulp.GUROBI_CMD(gapRel=gap, timeLimit=time_limit, keepFiles=False,
                                 logPath=os.path.join(self.config.base_dir, "my_demo", "output", "solver_log.txt"))
        self.model.solve(solver)

    def modify_org_map_df_by_solution(self):
        # todo unfinished
        feature_modify_mark = set()
        self.graph_error = 0
        for i, j_dict in self.y.items():
            for j, value in j_dict.items():
                if (i, j) in feature_modify_mark or (j, i) in feature_modify_mark:
                    continue
                if value.varValue > 0.99:
                    self.modify_df_arc_with_attr(i, j, "to_fe")
                else:
                    self.modify_df_arc_with_attr(i, j, "to_infe")
                feature_modify_mark.add((i, j))

        type_modify_mark = set()
        for i, j_dict in self.x.items():
            for j, value in j_dict.items():
                if (i, j) in type_modify_mark or (j, i) in type_modify_mark:
                    continue
                if value.varValue > 0.99:
                    self.modify_df_arc_with_attr(i, j, "change")
                    type_modify_mark.add((i, j))

    def modify_df_arc_with_attr(self, i, j, tag, attr_name=None):
        row = self.get_row_info_by_arc(i, j)
        modified_list = []

        if tag == "to_fe":
            if not row['curb_height_max_include']:
                row['curb_height_max'] = self.config.user_model["max_curb_height"]
                modified_list.append(f"{tag}_curb_height_max")
                self.graph_error += 1
            if not row['obstacle_free_width_float_include']:
                row['obstacle_free_width_float'] = self.config.user_model["min_sidewalk_width"]
                modified_list.append(f"{tag}_obstacle_free_width_float")
                self.graph_error += 1

        elif tag == "to_infe":
            if row['curb_height_max_include'] and row['obstacle_free_width_float_include']:
                # 因为 改变这个属性没有限制 改变高度需要判断路径类型
                row['obstacle_free_width_float'] = max(self.config.user_model["min_sidewalk_width"] - 1, 0)
                modified_list.append(f"{tag}_obstacle_free_width_float")
                self.graph_error += 1

        elif tag == "change":
            row["path_type"] = ("bike" if row["path_type"] == "walk" else "walk")
            modified_list.append(f"{tag}_path_type")
            self.graph_error += 1

        if row['modified'] is None:
            row['modified'] = modified_list
        else:
            row['modified'] = row['modified'] + modified_list

        self.data_holder.row_data.update({(i, j): row})
        self.org_map_df.loc[row.name] = row

    def fill_w_value_for_visual(self):
        self.org_map_df['w_i'] = 0
        self.org_map_df['w_j'] = 0
        self.org_map_df['y_ij'] = 0

        for idx, row in self.org_map_df.iterrows():
            arc = row['arc']
            self.org_map_df.at[row.name, "w_i"] = self.w[arc[0]].varValue
            self.org_map_df.at[row.name, "w_j"] = self.w[arc[1]].varValue
            self.org_map_df.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]].varValue

        self.df_path_foil['w_i'] = 0
        self.df_path_foil['w_j'] = 0
        self.df_path_foil['y_ij'] = 0

        cost = 0
        for idx, row in self.df_path_foil.iterrows():
            arc = row['arc']
            self.df_path_foil.at[row.name, "w_i"] = self.w[arc[0]].varValue
            self.df_path_foil.at[row.name, "w_j"] = self.w[arc[1]].varValue
            self.df_path_foil.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]].varValue
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["foil_cost"] = round(cost, 4)

        self.df_best_route['w_i'] = 0
        self.df_best_route['w_j'] = 0
        self.df_best_route['y_ij'] = 0
        cost = 0
        for idx, row in self.df_best_route.iterrows():
            arc = row['arc']
            self.df_best_route.at[row.name, "w_i"] = self.w[arc[0]].varValue
            self.df_best_route.at[row.name, "w_j"] = self.w[arc[1]].varValue
            self.df_best_route.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]].varValue
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["best_cost"] = round(cost, 4)

        self.df_path_fact['w_i'] = 0
        self.df_path_fact['w_j'] = 0
        self.df_path_fact['y_ij'] = 0
        cost = 0
        for idx, row in self.df_path_fact.iterrows():
            arc = row['arc']
            self.df_path_fact.at[row.name, "w_i"] = self.w[arc[0]].varValue
            self.df_path_fact.at[row.name, "w_j"] = self.w[arc[1]].varValue
            self.df_path_fact.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]].varValue
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["fact_cost"] = round(cost, 4)
        pass
