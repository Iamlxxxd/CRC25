# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 14:25
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
from my_demo.search.ArcModifyTag import ArcModifyTag
from my_demo.search.DataHolder import DataHolder
from my_demo.search.DataAnalyzer import DataAnalyzer
from my_demo.search.Operator import Operator
from my_demo.search.SubProblem import SubProblem
from router import Router
from utils.dataparser import create_network_graph, handle_weight, handle_weight_with_recovery
from utils.common_utils import set_seed, ensure_crs, correct_arc_direction, get_constraint_string, extract_nodes
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from my_demo.visual import visual_sub_problem, visual_map_foil_modded
from my_demo.search.TrackedCounter import TrackedCounter


class SearchSolver:
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

        self.analyzer = DataAnalyzer(self)
        self.operator_factory = Operator(self)

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
        self.process_org_map_df()
        self.process_foil()
        self.process_fact()

        self.current_solution_map = deepcopy(self.org_map_df)

    def process_org_map_df(self):
        self.org_map_df['org_obstacle_free_width_float'] = self.org_map_df['obstacle_free_width_float']
        self.org_map_df['org_curb_height_max'] = self.org_map_df['curb_height_max']
        self.handle_weight_without_preference(self.org_map_df)

        self.process_nodes()
        self.process_arcs()

        pass

    def process_nodes(self):
        all_points = pd.concat([self.org_map_df['geometry'].apply(lambda x: x.coords[0]),
                                self.org_map_df['geometry'].apply(lambda x: x.coords[1])]).drop_duplicates().tolist()
        point_id_map = {pt: idx for idx, pt in enumerate(all_points)}
        self.data_holder.all_nodes = list(point_id_map.values())
        self.data_holder.point_id_map = point_id_map
        self.data_holder.id_point_map = {v: k for k, v in point_id_map.items()}

        self.data_holder.start_node = self.data_holder.point_id_map.get(self.origin_node)
        self.data_holder.end_node = self.data_holder.point_id_map.get(self.dest_node)

    def process_arcs(self):
        self.data_holder.M = self.org_map_df['c'].sum()
        self.org_map_df["arc"] = None  # 用来表示边id
        self.org_map_df['modified'] = [[] for _ in range(len(self.org_map_df))]  # 用来表示是否被修改过

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
                if node1 != node2:
                    self.data_holder.all_arcs.append((node2, node1))

                for attr_name in self.data_holder.features:
                    ez_attr = self.eazy_name_map[attr_name]
                    if row[f"{attr_name}_include"]:
                        self.data_holder.all_feasible_both_way[ez_attr].append((node1, node2))

                        self.data_holder.all_feasible_arcs[ez_attr].append((node1, node2))
                        if node1 != node2:
                            self.data_holder.all_feasible_arcs[ez_attr].append((node2, node1))
                    else:
                        self.data_holder.all_infeasible_arcs[ez_attr].append((node1, node2))
                        if node1 != node2:
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
        self.df_path_foil['arc'] = self.df_path_foil['geometry'].apply(get_nodes)
        self.df_path_foil = correct_arc_direction(self.df_path_foil, self.data_holder.start_node,
                                                  self.data_holder.end_node)
        # self.df_path_foil.sort_values(by=['arc'], inplace=True)

        foil_cost = 0
        for idx, row in self.df_path_foil.iterrows():
            foil_cost += self.get_row_info_by_arc(row['arc'][0], row['arc'][1])['c']
        self.data_holder.foil_cost = foil_cost
        self.data_holder.foil_route_arcs = self.df_path_foil['arc'].tolist()

    def process_fact(self):
        get_nodes = lambda x: (self.data_holder.point_id_map.get(x.coords[0]),
                               self.data_holder.point_id_map.get(x.coords[1]))
        fact_cost = 0
        self.df_path_fact['arc'] = self.df_path_fact['geometry'].apply(get_nodes)
        self.df_path_fact = correct_arc_direction(self.df_path_fact, self.data_holder.start_node,
                                                  self.data_holder.end_node)
        for idx, row in self.df_path_fact.iterrows():
            fact_cost += self.get_row_info_by_arc(row['arc'][0], row['arc'][1])['c']
        self.data_holder.fact_cost = fact_cost

    def handle_weight_without_preference(self, df: GeoDataFrame):
        user_model = self.config.user_model

        # Don't include crossings with curbs that are too high
        df.loc[df['curb_height_max'] > user_model["max_curb_height"], 'include'] = 0

        # Don't include paths that are too n
        # arrow
        df.loc[df['obstacle_free_width_float'] < user_model["min_sidewalk_width"], 'include'] = 0

        # 高度是不是能改
        df['modify_able_curb_height_max'] = df["crossing_type"] == "curb_height"

        # 路径类型是不是能改
        df['modify_able_path_type'] = (df["path_type"] == "walk") | (df["path_type"] == "bike")

        # 高度不能改 或者满足条件
        df['curb_height_max_include'] = (df
                                         .apply(lambda x:
                                                (not x['modify_able_curb_height_max'])
                                                or (x['curb_height_max'] <= self.config.user_model["max_curb_height"]),
                                                axis=1
                                                )
                                         )

        df['obstacle_free_width_float_include'] = (df['obstacle_free_width_float']
                                                   .apply(lambda x: x >= self.config.user_model["min_sidewalk_width"]))

        df[["Deleta_p", "Deleta_n"]] = df.apply(
            lambda x: (0, 1) if x['curb_height_max_include'] and x['obstacle_free_width_float_include']
            else (2, 0) if not x['curb_height_max_include'] and not x['obstacle_free_width_float_include']
            else (1, 0),
            axis=1,
            result_type="expand"
        )

        # Define weight (combination of objectives)
        df['c'] = np.where(pd.isna(df['length']), 0, df['length'])
        df['d'] = 0

        df.loc[df['crossing'] == 'Yes', 'c'] = df['c'] * user_model["crossing_weight_factor"]

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

        df['my_weight'] = df['length']
        if user_model["walk_bike_preference"] == 'walk':
            df.loc[df['path_type'] == 'walk', 'my_weight'] = df['my_weight'] * user_model[
                "walk_bike_preference_weight_factor"]
        elif user_model["walk_bike_preference"] == 'bike':
            df.loc[df['path_type'] == 'bike', 'my_weight'] = df['my_weight'] * user_model[
                "walk_bike_preference_weight_factor"]

        df['my_weight'] = df['my_weight'] / df['my_weight'].abs().max()

    def get_row_info_by_arc(self, i, j):
        return self.data_holder.row_data.get((i, j), self.data_holder.row_data.get((j, i), None))

    def modify_df_arc_with_attr(self, i, j, tag: ArcModifyTag, attr_name=None):
        row = self.get_row_info_by_arc(i, j)
        modified_list = []

        if tag == ArcModifyTag.TO_FE:
            if not row['curb_height_max_include']:
                row['curb_height_max'] = self.config.user_model["max_curb_height"]
                modified_list.append(f"{tag.name}_curb_height_max")
            if not row['obstacle_free_width_float_include']:
                row['obstacle_free_width_float'] = self.config.user_model["min_sidewalk_width"]
                modified_list.append(f"{tag.name}_obstacle_free_width_float")

        elif tag == ArcModifyTag.TO_INFE:
            if row['curb_height_max_include'] and row['obstacle_free_width_float_include']:
                # 因为 改变这个属性没有限制 改变高度需要判断路径类型
                row['obstacle_free_width_float'] = max(self.config.user_model["min_sidewalk_width"] - 1, 0)
                modified_list.append(f"{tag.name}_obstacle_free_width_float")

        elif tag == ArcModifyTag.CHANGE:
            row["path_type"] = ("bike" if row["path_type"] == "walk" else "walk")
            modified_list.append(f"{tag.name}_path_type")

        if row['modified'] is None:
            row['modified'] = modified_list
        else:
            row['modified'] = row['modified'] + modified_list

        return row
        # self.data_holder.row_data.update({(i, j): row})
        # self.org_map_df.loc[row.name] = row

    def do_solve(self):
        self.analyzer.do_basic_analyze()
        # 这里把foil的不可行变成可行 存到了current_solution_map里
        self.operator_factory.do_foil_must_be_feasible()

        self.get_best_route_df_from_solution()
        route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                    self.config.user_model["attrs_variable_names"])
        if route_error <= 0:
            # 说明这张图只要把foil从不可行变成可行即可
            return

        counter = TrackedCounter(start=0, step=1)

        root_info = self.process_data_for_root_problem()
        root_problem = SubProblem(self, root_info, self.current_solution_map, self.new_graph, None, counter,0)
        self.solutions = root_problem.do_solve()

        # self.solutions = []
        # for fork, info in self.data_holder.foil_fact_fork_merge_nodes.items():
        #     sub_problem = SubProblem(self, info, self.current_solution_map, self.org_graph, None, counter)
        #     sub_solution = sub_problem.do_solve()
        #     self.solutions.extend(sub_solution)

        # self.operator_factory.do_must_be_infeasible_arcs()
        pass

    def process_solution_from_model(self):
        for sub_problem in self.solutions:
            for row in sub_problem.modified_row:
                self.current_solution_map.loc[row.name] = row
        self.get_best_route_df_from_solution()
        self.calc_error()

    def get_best_route_df_from_solution(self):
        self.current_solution_map = handle_weight_with_recovery(self.current_solution_map, self.config.user_model)

        _, self.new_graph = create_network_graph(self.current_solution_map)

        origin_node, dest_node, _, _, _ = self.router.set_o_d_coords(self.new_graph, self.config.gdf_coords_loaded)
        _, _, self.df_path_best = self.router.get_route(self.new_graph, origin_node, dest_node, self.heuristic_f)
        self.df_path_best = correct_arc_direction(self.df_path_best, self.data_holder.start_node,
                                                  self.data_holder.end_node)
        pass

    def calc_error(self):
        sub_op_list = get_virtual_op_list(self.org_map_df, self.current_solution_map,
                                          self.config.user_model["attrs_variable_names"])
        graph_error = len([op for op in sub_op_list if op[3] == "success"])

        route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                    self.config.user_model["attrs_variable_names"])

        self.data_holder.visual_detail_info['graph_error'] = graph_error
        self.data_holder.visual_detail_info['route_error'] = route_error
        self.data_holder.visual_detail_info['route_error_threshold'] = self.config.user_model["route_error_threshold"]

    def process_data_for_root_problem(self):
        # todo 把根节点当子问题 方便递归修正 未完成
        fork = self.data_holder.start_node
        merge = self.data_holder.end_node
        foil_sub_path = extract_nodes(self.df_path_foil)
        fact_sub_path = extract_nodes(self.df_path_best)

        # info = self.process_data_for_root_problem()
        # root_problem = SubProblem(self, info, self.current_solution_map, self.org_graph, None, counter)
        return {'fork': fork,
                'merge': merge,
                'foil_sub_path': foil_sub_path,
                'fact_sub_path': fact_sub_path}

    def process_visual_data(self) -> dict:

        return {"gdf_coords": self.config.gdf_coords_loaded,
                "origin_node_loc_length": self.origin_node_loc,
                "dest_node_loc_length": self.dest_node_loc,
                "meta_map": self.meta_map,
                "df_path_fact": self.df_path_fact,
                "df_path_foil": self.df_path_foil,
                "best_route": self.df_path_best,
                "org_map_df": self.current_solution_map,
                "config": self.config,
                "data_holder": self.data_holder,
                "show_data": self.data_holder.visual_detail_info}
