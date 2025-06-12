# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 10:35
@project:    CRC25
"""
from my_demo.config import Config
from copy import deepcopy
from typing import List
import multiprocessing
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from networkx.classes import DiGraph
from tqdm import tqdm
from collections import defaultdict
from my_demo.config import Config
from my_demo.mip.DataHolder import DataHolder
from my_demo.solver.CrossOperator import CrossOperator
from my_demo.solver.FitMeasurer import FitMeasurer
from my_demo.solver.Individual import Individual
from my_demo.solver.MutOperator import MutOperator
from my_demo.solver.PopInit.PopInitializer import PopInitializer
from my_demo.solver.SelectOperator import SelectOperator
from router import Router
from utils.dataparser import create_network_graph, handle_weight
from utils.common_utils import set_seed, ensure_crs
from gurobipy import *


class ModelSolver:
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

    eazy_name_map = {"curb_height_max": "H", "obstacle_free_width_float": "W"}
    eazy_name_map_reversed = {"H": "curb_height_max", "W": "obstacle_free_width_float"}

    def __init__(self, config: Config):
        self.config = config
        self.meta_map = config.meta_map
        self.data_holder = DataHolder()

        self.router = Router(heuristic=self.heuristic, CRS=self.meta_map["CRS"], CRS_map=self.meta_map["CRS_map"])
        self.load_basic_data()
        self.data_process()

    def load_basic_data(self):
        self.org_map_df = self.config.basic_network
        self.weight_df = deepcopy(self.org_map_df)
        self.weight_df = handle_weight(self.weight_df, self.config.user_model)
        self.weight_df_cp = deepcopy(self.weight_df)

        _, self.org_graph = create_network_graph(self.weight_df)

        self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, _ = self.router.set_o_d_coords(
            self.org_graph,
            self.config.gdf_coords_loaded)
        self.path_fact, self.G_path_fact, self.df_path_fact = self.router.get_route(self.org_graph, self.origin_node,
                                                                                    self.dest_node, self.heuristic_f)
        self.df_path_foil = self.config.df_path_foil

    def data_process(self):
        self.org_map_df_process()

    def org_map_df_process(self):
        self.handle_weight_without_preference(self.org_map_df)

        self.process_nodes()
        self.process_arcs()
        self.process_foil()
        pass

    def process_nodes(self):
        all_points = pd.concat([self.org_map_df['geometry'].apply(lambda x: x.coords[0]),
                                self.org_map_df['geometry'].apply(lambda x: x.coords[1])]).drop_duplicates().tolist()
        point_id_map = {pt: idx for idx, pt in enumerate(all_points)}
        self.data_holder.all_nodes = list(point_id_map.values())
        self.data_holder.point_id_map = point_id_map

    def process_arcs(self):
        self.data_holder.M = self.org_map_df['c'].sum()
        self.org_map_df["arc"] = None  # 用来表示边id
        self.org_map_df['modified'] = None  # 用来表示是否被修改过

        for idx, row in self.org_map_df.iterrows():
            point1 = row['geometry'].coords[0]
            node1 = self.data_holder.point_id_map.get(point1, None)
            point2 = row['geometry'].coords[1]
            node2 = self.data_holder.point_id_map.get(point2, None)
            self.org_map_df.at[row.name, "arc"] = (node1, node2)

            if node1 is None or node2 is None:
                # todo log
                print(point1, point2, "is None")
                continue

            if pd.isna(row['bikepath_id']):
                # 双向路
                self.data_holder.all_arcs.append((node1, node2))
                self.data_holder.all_arcs.append((node2, node1))

                for attr_name in self.data_holder.features:
                    ez_attr = self.eazy_name_map[attr_name]
                    if row[f"{attr_name}_include"]:
                        self.data_holder.all_feasible_both_way[ez_attr].append((node1, node2))

                        self.data_holder.all_feasible_arcs[ez_attr].append((node1, node2))
                        self.data_holder.all_feasible_arcs[ez_attr].append((node2, node1))
                    else:
                        self.data_holder.all_infeasible_arcs[ez_attr].append((node1, node2))
                        self.data_holder.all_infeasible_arcs[ez_attr].append((node2, node1))

                        self.data_holder.all_infeasible_both_way[ez_attr].append((node1, node2))
            else:
                self.data_holder.all_arcs.append((node1, node2))
                for attr_name in self.data_holder.features:
                    ez_attr = self.eazy_name_map[attr_name]
                    if row[f"{attr_name}_include"]:
                        self.data_holder.all_feasible_arcs[ez_attr].append((node1, node2))
                        self.data_holder.all_feasible_dir_arcs[ez_attr].append((node1, node2))
                    else:
                        self.data_holder.all_infeasible_arcs[ez_attr].append((node1, node2))
                        self.data_holder.all_infeasible_dir_arcs[ez_attr].append((node1, node2))

            self.data_holder.row_data[(node1, node2)] = self.org_map_df.loc[row.name].copy()

    def process_foil(self):
        get_nodes = lambda x: (self.data_holder.point_id_map.get(x.coords[0]),
                               self.data_holder.point_id_map.get(x.coords[1]))
        self.df_path_foil['arc'] = self.config.df_path_foil['geometry'].apply(get_nodes)
        self.df_path_foil.sort_values(by=['arc'], inplace=True)

        foil_cost = 0
        for idx, row in self.df_path_foil.iterrows():
            foil_cost += self.data_holder.row_data.get(row['arc'])['c']
        self.data_holder.foil_cost_without_preference = foil_cost
        self.data_holder.foil_route_arcs = self.df_path_foil['arc'].tolist()

    def handle_weight_without_preference(self, df: GeoDataFrame):
        user_model = self.config.user_model

        # Don't include crossings with curbs that are too high
        df.loc[df['curb_height_max'] > user_model["max_curb_height"], 'include'] = 0

        # Don't include paths that are too n
        # arrow
        df.loc[df['obstacle_free_width_float'] < user_model["min_sidewalk_width"], 'include'] = 0

        df['curb_height_max_include'] = (df['curb_height_max']
                                         .apply(lambda x: x <= self.config.user_model["max_curb_height"]))

        df['obstacle_free_width_float_include'] = (df['obstacle_free_width_float']
                                                   .apply(lambda x: x >= self.config.user_model["min_sidewalk_width"]))

        # Define weight (combination of objectives)
        df['c'] = df['length']

        df.loc[df['crossing'] == 'Yes', 'c'] = df['length'] * user_model["crossing_weight_factor"]

        coef_perferred = ((1 - user_model["walk_bike_preference_weight_factor"]) / user_model[
            "walk_bike_preference_weight_factor"])
        coef_unperferred = user_model["walk_bike_preference_weight_factor"] - 1

        # todo 可能有数值问题 会导致无解?
        if user_model["walk_bike_preference"] == 'walk':
            df.loc[df['path_type'] == 'walk', 'c'] = (df['c'] * user_model["walk_bike_preference_weight_factor"])

            df.loc[df['path_type'] == 'walk', 'd'] = df['c'] * coef_perferred
            df.loc[df['path_type'] == 'bike', 'd'] = df['c'] * coef_unperferred
        elif user_model["walk_bike_preference"] == 'bike':
            df.loc[df['path_type'] == 'bike', 'c'] = df['c'] * user_model["walk_bike_preference_weight_factor"]

            df.loc[df['path_type'] == 'bike', 'd'] = df['c'] * coef_perferred
            df.loc[df['path_type'] == 'walk', 'd'] = df['c'] * coef_unperferred

        # if user_model["walk_bike_preference"] == 'walk':
        #     df.loc[df['path_type'] == 'walk', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        # elif user_model["walk_bike_preference"] == 'bike':
        #     df.loc[df['path_type'] == 'bike', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        #
        # df['my_weight'] = df['my_weight'] /df['my_weight'].abs().max()

    def get_row_info_by_arc(self, i, j):
        return self.data_holder.row_data.get((i, j), self.data_holder.row_data.get((j, i), None))

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
        self.w = self.model.addVars(self.data_holder.all_nodes, vtype=GRB.CONTINUOUS, name="w_")
        big_m = self.data_holder.M

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
                self.model.addConstr(self.x_pos[f, i, j] + self.x_pos[f, j, i] <= 1, name="ua_pos_[{},{}]".format(i, j))
                # self.model.addConstr(self.x_neg[f, i, j] + self.x_neg[f, j, i] <= 1, name="ua_neg_[{},{}]".format(i, j))
                self.model.addConstr(self.x_p[i, j] + self.x_p[j, i] <= 1, name="ua_p_[{},{}]".format(i, j))
                self.model.addConstr(self.y[i, j] + self.y[j, i] <= 1, name="ua_y_[{},{}]".format(i, j))

        for f, arcs in self.data_holder.all_infeasible_both_way.items():
            for i, j in arcs:
                self.model.addConstr(self.x_neg[f, i, j] + self.x_neg[f, j, i] <= 1, name="ua_neg_[{},{}]".format(i, j))
                # self.model.addConstr(self.x_pos[f, i, j] + self.x_pos[f, j, i] <= 1, name="ua_pos_[{},{}]".format(i, j))
                self.model.addConstr(self.x_p[i, j] + self.x_p[j, i] <= 1, name="ua_p_[{},{}]".format(i, j))
                self.model.addConstr(self.y[i, j] + self.y[j, i] <= 1, name="ua_y_[{},{}]".format(i, j))

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

        # todo 会让一些无关紧要的变量也赋值
        # x_pos_sum = quicksum((self.x_pos[k, i, j] for k in self.data_holder.all_feasible_arcs
        #                       for (i, j) in self.data_holder.all_feasible_arcs[k]))
        # x_neg_sum = quicksum((self.x_neg[k, i, j] for k in self.data_holder.all_infeasible_arcs
        #                       for (i, j) in self.data_holder.all_infeasible_arcs[k]))
        # x_p_sum = self.x_p.sum()
        # self.model.setObjective(x_pos_sum + x_neg_sum + x_p_sum, GRB.MINIMIZE)
        self.model.setObjective(self.x_pos.sum() + self.x_neg.sum() + self.x_p.sum(), GRB.MINIMIZE)

    def solve_model(self, time_limit=3600, gap=0.01):
        self.model.setParam(GRB.Param.MIPGap, gap)
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.update()
        self.model.optimize()

    def process_solution_from_model(self):
        self.modify_org_map_df_by_solution()
        self.get_best_route_df_from_solution()

    def get_best_route_df_from_solution(self):
        selected_rows = []
        for (i, j), value in self.y.items():
            if value.X > 0.9:
                row = self.get_row_info_by_arc(i, j)
                selected_rows.append(row)

        if not selected_rows:
            return None
        self.df_best_route = GeoDataFrame(selected_rows, geometry='geometry', crs=self.org_map_df.crs)
        self.df_best_route.sort_values(by=['arc'], inplace=True)

    def modify_org_map_df_by_solution(self):

        modify_dict = defaultdict(list)
        self.graph_error = 0
        for (f, i, j), value in self.x_pos.items():
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(attr_name, i, j)

                if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                    continue
                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (f, i, j), value in self.x_neg.items():
            if value.X > 0.99:
                attr_name = self.eazy_name_map_reversed.get(f)
                self.modify_df_arc_with_attr(attr_name, i, j)

                if ((f, i, j) in modify_dict) or ((f, j, i) in modify_dict):
                    continue
                self.graph_error += 1
                modify_dict[(f, i, j)].append(attr_name)

        for (i, j), value in self.x_p.items():
            if value.X > 0.99:
                self.modify_df_arc_with_attr("path_type", i, j)

                if ((i, j) in modify_dict) or ((j, i) in modify_dict):
                    continue
                self.graph_error += 1
                modify_dict[(i, j)].append("path_type")

        pass

    def process_visual_data(self) -> dict:

        return {"gdf_coords": self.config.gdf_coords_loaded,
                "origin_node_loc_length": self.origin_node_loc,
                "dest_node_loc_length": self.dest_node_loc,
                "meta_map": self.meta_map,
                "df_path_fact": self.df_path_fact,
                "df_path_foil": self.df_path_foil,
                "best_route": self.df_best_route,
                "org_map_df": self.org_map_df,
                "config": self.config,
                "data_holder": self.data_holder}

    def modify_df_arc_with_attr(self, attr_name, i, j):
        row = self.get_row_info_by_arc(i, j)

        if attr_name == "curb_height_max":
            # 需要修改高度
            row[attr_name] = self.config.user_model["max_curb_height"]
        elif attr_name == "obstacle_free_width_float":
            row[attr_name] = self.config.user_model["min_sidewalk_width"]
        elif attr_name == "path_type":
            row[attr_name] = ("bike" if row[attr_name] == "walk" else "walk")

        if row['modified'] is None:
            row['modified'] = [attr_name]
        else:
            row['modified'].append(attr_name)

        self.data_holder.row_data.update({(i, j): row})
        self.org_map_df.loc[row.name] = row
