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
from utils.common_utils import set_seed
from gurobipy import *


class ModelSolver:
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

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
        self.org_map_df['start_point'] = self.org_map_df['geometry'].apply(lambda x: x.coords[0])
        self.org_map_df['end_point'] = self.org_map_df['geometry'].apply(lambda x: x.coords[-1])

        all_points = pd.concat(
            [self.org_map_df['start_point'], self.org_map_df['end_point']]).drop_duplicates().tolist()
        point_id_map = {pt: idx for idx, pt in enumerate(all_points)}
        self.org_map_df['start_id'] = self.org_map_df['start_point'].map(point_id_map)
        self.org_map_df['end_id'] = self.org_map_df['end_point'].map(point_id_map)
        self.org_map_df['edge'] = list(zip(self.org_map_df['start_id'], self.org_map_df['end_id']))

        self.data_holder.all_nodes = all_points
        self.data_holder.all_feasible_edges = self.org_map_df.loc[self.org_map_df['include'] == 1]['edge'].tolist()
        self.data_holder.all_infeasible_edges = self.org_map_df.loc[self.org_map_df['include'] == 0]['edge'].tolist()
        self.data_holder.all_edges = self.org_map_df['edge'].tolist()

        # 生成 in_degree_dict, out_degree_dict, edge_dict
        self.data_holder.in_degree_dict = self.org_map_df.groupby('end_id')['start_id'].agg(list).to_dict()
        self.data_holder.out_degree_dict = self.org_map_df.groupby('start_id')['end_id'].agg(list).to_dict()
        self.data_holder.edge_dict = self.org_map_df.set_index('edge').to_dict(orient='index')

        #todo
        temp = pd.merge(self.config.df_path_foil, self.org_map_df[['geometry','edge']], how='left', on=['geometry']).sort_values('edge')

        pass

    def handle_weight_without_preference(self, df: GeoDataFrame):
        user_model = self.config.user_model
        # Don't include crossings with curbs that are too high
        df.loc[df['curb_height_max'] > user_model["max_curb_height"], 'include'] = 0

        # Don't include paths that are too n
        # arrow
        df.loc[df['obstacle_free_width_float'] < user_model["min_sidewalk_width"], 'include'] = 0

        # Define weight (combination of objectives)
        df['my_weight'] = df['length']

        df.loc[df['crossing'] == 'Yes', 'my_weight'] = df['length'] * user_model["crossing_weight_factor"]

        # if user_model["walk_bike_preference"] == 'walk':
        #     df.loc[df['path_type'] == 'walk', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        # elif user_model["walk_bike_preference"] == 'bike':
        #     df.loc[df['path_type'] == 'bike', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        #
        # df['my_weight'] = df['my_weight'] /df['my_weight'].abs().max()

    def init_model(self):
        model = Model('CRC25')

        x_pos = model.addVars(self.data_holder.all_feasible_edges, vtype=GRB.BINARY, name="xPos_")
        x_neg = model.addVars(self.data_holder.all_infeasible_edges, vtype=GRB.BINARY, name="xNeg_")
        x_p = model.addVars(self.data_holder.all_edges, vtype=GRB.BINARY, name="xP_")
        y = model.addVars(self.data_holder.all_edges, vtype=GRB.BINARY, name="y_")
        w = model.addVars(self.data_holder.all_edges, vtype=GRB.BINARY, name="w_")

        # todo start and end
        # Flow Balance
        model.addConstrs((y.sum("*", i) - y.sum(i, "*") == 0 for i in self.data_holder.all_nodes), name="fb_")

        # Arc Feasibility Constraints
        for arc in self.data_holder.all_feasible_edges:
            model.addConstr(y[arc] <= 1 - x_pos[arc], name="arc_f_pos_{}".format(arc))

        for arc in self.data_holder.all_infeasible_edges:
            model.addConstr(y[arc] <= x_neg[arc], name="arc_f_neg_{}".format(arc))

        # Linearization Constraints
        model.addConstrs((w[arc] <= y[arc] for arc in self.data_holder.all_edges), name="w_y_")
        model.addConstrs((w[arc] <= x_p[arc] for arc in self.data_holder.all_edges), name="w_xP_")
        model.addConstrs((w[arc] >= y[arc] + x_p[arc] - 1 for arc in self.data_holder.all_edges), name="w_y_xP_")

        #todo foil path

