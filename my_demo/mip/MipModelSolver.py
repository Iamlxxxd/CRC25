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
                                self.org_map_df['geometry'].apply(lambda x: x.coords[-1])]).drop_duplicates().tolist()
        point_id_map = {pt: idx for idx, pt in enumerate(all_points)}
        self.data_holder.all_nodes = all_points
        self.data_holder.point_id_map = point_id_map

    def process_arcs(self):
        self.data_holder.M = self.org_map_df['c'].sum()

        for idx, row in self.org_map_df.iterrows():
            point1 = row['geometry'].coords[0]
            node1 = self.data_holder.point_id_map.get(point1, None)
            point2 = row['geometry'].coords[1]
            node2 = self.data_holder.point_id_map.get(point2, None)

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
                    if self.judge_arc_include(attr_name, row):
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
                    if self.judge_arc_include(attr_name, row):
                        self.data_holder.all_feasible_arcs[ez_attr].append((node1, node2))
                    else:
                        self.data_holder.all_infeasible_arcs[ez_attr].append((node1, node2))

            self.data_holder.row_data[(node1, node2)] = row

    def process_foil(self):
        get_nodes = lambda x: (self.data_holder.point_id_map.get(x.coords[0]),
                               self.data_holder.point_id_map.get(x.coords[1]))
        self.config.df_path_foil['arc'] = self.config.df_path_foil['geometry'].apply(get_nodes)
        foil_cost = 0
        for idx, row in self.config.df_path_foil.iterrows():
            foil_cost += self.data_holder.row_data.get(row['arc'])['my_weight']
        self.data_holder.foil_cost_without_preference = foil_cost
        self.data_holder.foil_route_arc = self.config.df_path_foil['arc'].tolist()

    def judge_arc_include(self, attr_name, row) -> bool:
        if attr_name == "curb_height_max":
            return row[attr_name] <= self.config.user_model["max_curb_height"]
        elif attr_name == "obstacle_free_width_float":
            return row[attr_name] >= self.config.user_model["min_sidewalk_width"]
        return False

    def handle_weight_without_preference(self, df: GeoDataFrame):
        user_model = self.config.user_model
        # Don't include crossings with curbs that are too high
        df.loc[df['curb_height_max'] > user_model["max_curb_height"], 'include'] = 0

        # Don't include paths that are too n
        # arrow
        df.loc[df['obstacle_free_width_float'] < user_model["min_sidewalk_width"], 'include'] = 0

        # Define weight (combination of objectives)
        df['c'] = df['length']

        df.loc[df['crossing'] == 'Yes', 'c'] = df['length'] * user_model["crossing_weight_factor"]

        # todo 可能有数值问题 会导致无解?
        if user_model["walk_bike_preference"] == 'walk':
            df.loc[df['path_type'] == 'walk', 'c'] = (df['c']
                                                      * user_model["walk_bike_preference_weight_factor"])

            df.loc[df['path_type'] == 'walk', 'd'] = (df['c']
                                                      * ((1 - user_model["walk_bike_preference_weight_factor"]) /
                                                         user_model["walk_bike_preference_weight_factor"]))
            df.loc[df['path_type'] == 'bike', 'd'] = (df['c']
                                                      * (user_model["walk_bike_preference_weight_factor"] - 1))
        elif user_model["walk_bike_preference"] == 'bike':
            df.loc[df['path_type'] == 'bike', 'd'] = (df['c']
                                                      * user_model["walk_bike_preference_weight_factor"])

            df.loc[df['path_type'] == 'bike', 'd'] = (df['c']
                                                      * ((1 - user_model["walk_bike_preference_weight_factor"]) /
                                                         user_model["walk_bike_preference_weight_factor"]))
            df.loc[df['path_type'] == 'walk', 'd'] = (df['c']
                                                      * (user_model["walk_bike_preference_weight_factor"] - 1))

        # if user_model["walk_bike_preference"] == 'walk':
        #     df.loc[df['path_type'] == 'walk', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        # elif user_model["walk_bike_preference"] == 'bike':
        #     df.loc[df['path_type'] == 'bike', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        #
        # df['my_weight'] = df['my_weight'] /df['my_weight'].abs().max()

    def init_model(self):
        model = Model('CRC25')

        x_pos = model.addVars(((k, i, j) for k in self.data_holder.all_feasible_arcs
                               for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="xPos_")

        x_neg = model.addVars(((k, i, j) for k in self.data_holder.all_infeasible_arcs
                               for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="xNeg_")

        x_p = model.addVars(((i, j) for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="xP_")
        y = model.addVars(((i, j) for (i, j) in self.data_holder.all_arcs), vtype=GRB.BINARY, name="y_")
        w = model.addVars(self.data_holder.all_nodes, vtype=GRB.CONTINUOUS, name="w_")
        big_m = self.data_holder.M

        # Arc Feasibility Constraints
        for f, arcs in self.data_holder.all_feasible_arcs:
            for i, j in arcs:
                model.addConstr(y[i, j] <= 1 - x_pos[f, i, j], name="arc_f_pos_[{},{},{}]".format(f, i, j))

        for f, arcs in self.data_holder.all_feasible_arcs:
            for i, j in arcs:
                model.addConstr(y[i, j] <= x_neg[f, i, j], name="arc_f_neg_[{},{},{}]".format(f, i, j))

        for i, j in self.data_holder.all_arcs:
            f_neg = quicksum(x_neg[k, i, j] for k in self.data_holder.all_infeasible_arcs)
            f_pos = quicksum(1 - x_neg[k, i, j] for k in self.data_holder.all_feasible_arcs)
            model.addConstr(y[i, j] >= f_neg + f_pos - (len(self.eazy_name_map) - 1),
                            name="arc_f_multiple_[{},{}]".format(i, j))

        # Undirected Arc Constraint
        for f, arcs in self.data_holder.all_feasible_both_way:
            for i, j in arcs:
                model.addConstr(x_pos[f, i, j] + x_pos[f, j, i] <= 1, name="ua_pos_[{},{}]".format(i, j))
                model.addConstr(x_p[i, j] + x_p[j, i] <= 1, name="ua_p_[{},{}]".format(i, j))
                model.addConstr(y[i, j] + y[j, i] <= 1, name="ua_y_[{},{}]".format(i, j))

        for f, arcs in self.data_holder.all_infeasible_both_way:
            for i, j in arcs:
                model.addConstr(x_neg[f, i, j] + x_neg[f, j, i] <= 1, name="ua_neg_[{},{}]".format(i, j))
                model.addConstr(x_p[i, j] + x_p[j, i] <= 1, name="ua_p_[{},{}]".format(i, j))
                model.addConstr(y[i, j] + y[j, i] <= 1, name="ua_y_[{},{}]".format(i, j))

        # Shortest Path Constraints
        for i, j in self.data_holder.all_arcs:
            row_data = self.get_row_info_by_arc(i, j)

            model.addConstr(w[j] - w[i] <= row_data['c'] + row_data['d'] * x_p[i, j] + big_m * (1 - y[i, j]),
                            name="sp_opt_[{},{}]".format(i, j))
        for i, j in self.data_holder.foil_route_arcs:
            row_data = self.get_row_info_by_arc(i, j)
            model.addConstr(w[j] - w[i] >= row_data['c'] + row_data['d'] * x_p[i, j] - big_m * (1 - y[i, j]),
                            name="sp_opt_[{},{}]".format(i, j))

            model.addConstr(y[i, j] == 1, "foil_y_[{},{}]".format(i, j))
        x_pos_sum = quicksum((x_pos[k, i, j] for k in self.data_holder.all_feasible_arcs
                               for (i, j) in self.data_holder.all_feasible_arcs[k]))
        x_neg_sum = quicksum((x_neg[k, i, j] for k in self.data_holder.all_infeasible_arcs
                               for (i, j) in self.data_holder.all_infeasible_arcs[k]))
        x_p_sum = x_p.sum()
        model.setObjective(x_pos_sum +x_neg_sum+x_p_sum, GRB.MINIMIZE)
        model.setParam(GRB.Param.MIPGap, 0.05)
        model.setParam(GRB.Param.TimeLimit, 3600)

        model.update()
        model.optimize()

        self.x_pos = x_pos
        self.x_neg = x_neg
        self.x_p = x_p
        self.y = y
    def get_row_info_by_arc(self, i, j):
        return self.data_holder.row_data.get((i, j), self.data_holder.row_data.get((i, j), None))
