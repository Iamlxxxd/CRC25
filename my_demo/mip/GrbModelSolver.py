#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/29 12:28
# @Author  : JunhaoShi(01387247)
# @Desc    :


from collections import defaultdict
from my_demo.config import Config
from my_demo.mip.ModelSolver import ModelSolver
from gurobipy import *


class GrbModelSolver(ModelSolver):

    def __init__(self, config: Config):
        super().__init__()

    def init_model(self):
        self.model = Model('CRC25')

        self.x_pos = self.model.addVars(((k, i, j) for k in self.data_holder.all_feasible_arcs
                                         for (i, j) in self.data_holder.all_feasible_arcs[k]), vtype=GRB.BINARY,
                                        name="xPos_")

        self.x_neg = self.model.addVars(((k, i, j) for k in self.data_holder.all_infeasible_arcs
                                         for (i, j) in self.data_holder.all_infeasible_arcs[k]), vtype=GRB.BINARY,
                                        name="xNeg_")

        self.x_p = self.model.addVars(((i, j) for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="xP_")
        self.y = self.model.addVars(((i, j) for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="y_")
        self.w = self.model.addVars(self.data_holder.all_nodes, vtype=GRB.CONTINUOUS, lb=0, name="w_")
        big_m = self.data_holder.M * 4

        # Arc Feasibility Constraints
        for f, arcs in self.data_holder.all_feasible_arcs.items():
            for i, j in arcs:
                self.model.addConstr(self.y[i, j] <= 1 - self.x_pos[f, i, j],
                                     name="arc_f_pos_[{},{},{}]".format(f, i, j))

        for f, arcs in self.data_holder.all_infeasible_arcs.items():
            for i, j in arcs:
                self.model.addConstr(self.y[i, j] <= self.x_neg[f, i, j], name="arc_f_neg_[{},{},{}]".format(f, i, j))

        for i, j in self.data_holder.all_arcs:
            # 仅在变量存在时才参与 quicksum
            f_neg = quicksum(
                self.x_neg[k, i, j]
                for k in self.data_holder.all_infeasible_arcs
                if (k, i, j) in self.x_neg
            )
            f_pos = quicksum(
                (1 - self.x_pos[k, i, j])
                for k in self.data_holder.all_feasible_arcs
                if (k, i, j) in self.x_pos
            )
            self.model.addConstr(self.y[i, j] >= f_neg + f_pos - (len(self.eazy_name_map) - 1),
                                 name="arc_f_multiple_[{},{}]".format(i, j))

        # Undirected Arc Constraint
        for f, arcs in self.data_holder.all_feasible_both_way.items():
            for i, j in arcs:
                self.model.addConstr(self.x_pos[f, i, j] == self.x_pos[f, j, i], name="ua_pos_[{},{}]".format(i, j))
                # self.model.addConstr(self.x_neg[f, i, j] == self.x_neg[f, j, i], name="ua_neg_[{},{}]".format(i, j))
                self.model.addConstr(self.x_p[i, j] == self.x_p[j, i], name="ua_p_[{},{}]".format(i, j))
                self.model.addConstr(self.y[i, j] == self.y[j, i], name="ua_y_[{},{}]".format(i, j))

        for f, arcs in self.data_holder.all_infeasible_both_way.items():
            for i, j in arcs:
                self.model.addConstr(self.x_neg[f, i, j] == self.x_neg[f, j, i], name="ua_neg_[{},{}]".format(i, j))
                # self.model.addConstr(self.x_pos[f, i, j] + self.x_pos[f, j, i] <= 1, name="ua_pos_[{},{}]".format(i, j))
                self.model.addConstr(self.x_p[i, j] == self.x_p[j, i], name="ua_p_[{},{}]".format(i, j))
                self.model.addConstr(self.y[i, j] == self.y[j, i], name="ua_y_[{},{}]".format(i, j))

        # Shortest Path Constraints
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model.addConstr(
                self.w[j] - self.w[i] <= row_data['c'] + row_data['d'] * self.x_p[i, j] + big_m * (1 - self.y[i, j]),
                name="sp_opt_[{},{}]".format(i, j))

        for i, j in self.data_holder.foil_route_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            self.model.addConstr(
                self.w[j] - self.w[i] >= row_data['c'] + row_data['d'] * self.x_p[i, j] - big_m * (1 - self.y[i, j]),
                name="sp_foil_[{},{}]".format(i, j))

            self.model.addConstr(self.y[i, j] == 1, "foil_y_[{},{}]".format(i, j))

        # self.model.addConstr(self.w[self.data_holder.start_node] == 0, "w_start")
        # self.model.addConstr(self.w[self.data_holder.end_node] == self.data_holder.foil_cost, "w_end")
        # todo 会让一些无关紧要的变量也赋值
        # x_pos_sum = quicksum((self.x_pos[k, i, j] for k in self.data_holder.all_feasible_arcs
        #                       for (i, j) in self.data_holder.all_feasible_arcs[k]))
        # x_neg_sum = quicksum((self.x_neg[k, i, j] for k in self.data_holder.all_infeasible_arcs
        #                       for (i, j) in self.data_holder.all_infeasible_arcs[k]))
        # x_p_sum = self.x_p.sum()
        # self.model.setObjective(x_pos_sum + x_neg_sum + x_p_sum, GRB.MINIMIZE)
        self.model.setObjective(self.x_pos.sum() + self.x_neg.sum() + self.x_p.sum(),
                                GRB.MINIMIZE)

    def solve_model(self, time_limit=3600, gap=0.01):
        self.model.setParam(GRB.Param.MIPGap, gap)
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.update()
        # debug
        # self.model.write("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/CRC25.lp")
        self.model.optimize()

        if self.model.status == 3 or self.model.status == 4 or self.model.status == 5:
            self.print_infeasible(self.model)

    def fill_w_value_for_visual(self):
        self.org_map_df['w_i'] = 0
        self.org_map_df['w_j'] = 0
        self.org_map_df['y_ij'] = 0
        for idx, row in self.org_map_df.iterrows():
            arc = row['arc']
            self.org_map_df.at[row.name, "w_i"] = self.w[arc[0]].X
            self.org_map_df.at[row.name, "w_j"] = self.w[arc[1]].X
            self.org_map_df.at[row.name, "y_ij"] = self.y[arc].X
        self.df_path_foil['w_i'] = 0
        self.df_path_foil['w_j'] = 0
        self.df_path_foil['y_ij'] = 0
        for idx, row in self.df_path_foil.iterrows():
            arc = row['arc']
            self.df_path_foil.at[row.name, "w_i"] = self.w[arc[0]].X
            self.df_path_foil.at[row.name, "w_j"] = self.w[arc[1]].X
            self.df_path_foil.at[row.name, "y_ij"] = self.y[arc].X

    def modify_org_map_df_by_solution(self):

        modify_dict = defaultdict(list)
        self.graph_error = 0
        for (f, i, j), value in self.x_pos.items():
            if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                continue
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(i, j, "to_infe", attr_name)

                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (f, i, j), value in self.x_neg.items():
            if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                continue
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(i, j, "to_fe", attr_name)

                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (i, j), value in self.x_p.items():
            if ((i, j) in modify_dict) or ((j, i) in modify_dict):
                continue
            if value.X > 0.99:
                self.modify_df_arc_with_attr(i, j, "change", "path_type")

                self.graph_error += 1
                modify_dict[(i, j)].append("path_type")

        pass

    def modify_df_arc_with_attr(self, i, j, tag, attr_name):
        row = self.get_row_info_by_arc(i, j)

        if attr_name == "curb_height_max":
            if tag == "to_fe":
                # 需要修改高度
                row[attr_name] = self.config.user_model["max_curb_height"]
            elif tag == "to_infe":
                row[attr_name] = self.config.user_model["max_curb_height"] + 1
        elif attr_name == "obstacle_free_width_float":
            if tag == "to_fe":
                row[attr_name] = self.config.user_model["min_sidewalk_width"]
            elif tag == "to_infe":
                row[attr_name] = max(self.config.user_model["min_sidewalk_width"] - 1, 0)
        elif attr_name == "path_type":
            row[attr_name] = ("bike" if row[attr_name] == "walk" else "walk")

        if row['modified'] is None:
            row['modified'] = [f"{tag}_{attr_name}"]
        else:
            row['modified'].append(f"{tag}_{attr_name}")

        self.data_holder.row_data.update({(i, j): row})
        self.org_map_df.loc[row.name] = row
