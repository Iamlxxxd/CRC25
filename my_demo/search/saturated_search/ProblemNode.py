# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/25 10:56
@project:    CRC25
"""
from copy import deepcopy
from typing import Optional

import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
import networkx as nx
from my_demo.search.DataHolder import DataHolder
from my_demo.search.saturated_search.MoveData import MoveData
from utils.dataparser import handle_weight_with_recovery, create_network_graph
import shapely.ops as so
import shapely.geometry as sg
import geopandas as gpd
import pandas as pd
import networkx as nx
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from utils.common_utils import correct_arc_direction
from my_demo.visual import visual_sub_problem, visual_map_foil_modded
from typing import List
from my_demo.search.ArcModifyTag import ArcModifyTag
from collections import defaultdict
from my_demo.search.saturated_search.Operator import add_one_move


class ProblemNode:

    def __init__(self, solver, info_tuple, modified_move_list: List[MoveData], map_df, map_graph, master, idx_gen,
                 level):
        self.idx_gen = idx_gen
        self.idx = next(idx_gen)
        self.level = level

        self.org_solver = solver
        self.config = solver.config
        self.data_holder = DataHolder()
        self.info_tuple = info_tuple

        self.fork = info_tuple['fork']
        self.merge = info_tuple['merge']

        self.sub_fact = info_tuple['fact_sub_path']
        self.sub_foil = info_tuple['foil_sub_path']

        self.map_df = deepcopy(map_df)
        # todo 暂时没操作图 先不deepcopy

        self.master = master

        self.map_graph = map_graph
        if master:
            # 用来分析 哪个算子生效多的 #todo 可能影响效率后面需要删除
            self.inherit_for_visual = deepcopy(master.inherit_for_visual)

            # 用来hash 认为操作一致的
            self.inherit = [] + master.inherit

        else:
            # 初始化嵌套字典结构
            self.inherit_for_visual = dict()
            self.inherit = []

            # 添加修改记录
        for move in modified_move_list:
            add_one_move(self.inherit_for_visual, move)

            self.inherit.append(move)

        self.modified_move_list = modified_move_list

        self.inherit = sorted(set(self.inherit))

        self.df_path_best: GeoDataFrame = None
        self.df_path_foil = self.org_solver.df_path_foil
        self.graph_error = 0
        self.route_error = 0

        self.move_str = None

    def apply_modified_arc(self):
        # todo 可以改成操作graph
        for move_data in self.modified_move_list:
            (i, j), modify_tag, name = move_data.arc, move_data.modify_tag, move_data.operator_type
            modified_row = self.org_solver.modify_df_arc_with_attr(i, j, modify_tag)
            solution_row = self.org_solver.current_solution_map.loc[modified_row.name]
            modified_row['modified'] = modified_row['modified'] + solution_row['modified']
            self.map_df.loc[modified_row.name] = modified_row

    def calc_error(self):
        sub_op_list = get_virtual_op_list(self.org_solver.org_map_df, self.map_df,
                                          self.config.user_model["attrs_variable_names"])
        self.graph_error = len([op for op in sub_op_list if op[3] == "success"])

        self.actual_route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                                self.config.user_model[
                                                                                    "attrs_variable_names"])

        self.route_error = max(self.actual_route_error - self.config.user_model["route_error_threshold"], 0)

    def calc_sub_best(self):
        self.weight_df = handle_weight_with_recovery(self.map_df, self.config.user_model)

        _, self.new_graph = create_network_graph(self.weight_df)

        # self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, _ = self.set_o_d_coords(
        #     origin_lc,
        #     dest_lc,
        #     self.org_graph)
        self.path_best, self.G_path_best, self.df_path_best = self.org_solver.router.get_route(self.new_graph,
                                                                                               self.org_solver.data_holder.start_node_lc,
                                                                                               self.org_solver.data_holder.end_node_lc,
                                                                                               self.org_solver.heuristic_f)
        self.df_path_best = correct_arc_direction(self.df_path_best, self.org_solver.data_holder.start_node_id,
                                                  self.org_solver.data_holder.end_node_id)

    def __lt__(self, other):
        # 先比route_error，再比graph_error
        # return (-self.level, self.route_error, self.graph_error) < (-self.level, other.route_error, other.graph_error)
        return (self.route_error, self.graph_error) < (other.route_error, other.graph_error)

    def __hash__(self):
        return hash(str(self.inherit))

    def __eq__(self, other):
        if not isinstance(other, ProblemNode):
            return False
        return str(self.inherit) == str(other.inherit)

    def __str__(self):
        return f"【{self.idx},{self.level}:r:{self.route_error}g:{self.graph_error}】#【{self.fork}#{self.merge}#{self.visual_move_data()}】"

    def __repr__(self):
        return self.__str__()

    def visual_move_data(self) -> str:
        if self.move_str is not None:
            return self.move_str

        return_str = ""
        for move in self.modified_move_list:
            return_str += f"{move.arc},{move.modify_tag.name},{move.operator_type}"

        self.move_str = f"【{return_str}】"
        return self.move_str

    def better_than_other(self, other):
        return (self.route_error, self.graph_error) < (other.route_error, other.graph_error)
