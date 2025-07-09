# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/25 10:56
@project:    CRC25
"""
import os
import sys
from copy import deepcopy
from geopandas import GeoDataFrame
import geopandas as gpd
import pandas as pd
import numpy as np
from queue import PriorityQueue
import time
import random
import math
from shapely import to_wkt

from config import Config
from src.DataHolder import DataHolder
from src.calc.router import Router
from src.calc.DataAnalyzer import DataAnalyzer
from src.solver.ProblemNode import ProblemNode
from src.calc.dataparser import handle_weight, handle_weight_with_recovery, create_network_graph
from src.calc.common_utils import correct_arc_direction, extract_nodes, edge_betweenness_to_target_multigraph
from src.solver.ArcModifyTag import ArcModifyTag
from src.calc.metrics import get_virtual_op_list, common_edges_similarity_route_df_weighted
from src.TrackedCounter import TrackedCounter
from src.solver.Operator import do_foil_must_be_feasible, generate_multi_modify_arc_by_graph_feature
from src.calc.dataparser import convert


class SearchSolverSaturated:
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

        self.best_leaf_node: ProblemNode = None
        self.current_best: ProblemNode = None

    def load_basic_data(self):
        self.org_map_df = self.config.basic_network
        self.org_df_from_io = deepcopy(self.org_map_df)
        self.org_df_from_io = handle_weight(self.org_df_from_io, self.config.user_model)

        _, self.org_graph = create_network_graph(self.org_df_from_io)

        self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, _ = self.router.set_o_d_coords(
            self.org_graph,
            self.config.gdf_coords_loaded)

        self.data_holder.start_node_lc = self.origin_node
        self.data_holder.end_node_lc = self.dest_node

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

        self.data_holder.start_node_id = self.data_holder.point_id_map.get(self.origin_node)
        self.data_holder.end_node_id = self.data_holder.point_id_map.get(self.dest_node)

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
        self.df_path_foil = correct_arc_direction(self.df_path_foil, self.data_holder.start_node_id,
                                                  self.data_holder.end_node_id)
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
        self.df_path_fact = correct_arc_direction(self.df_path_fact, self.data_holder.start_node_id,
                                                  self.data_holder.end_node_id)
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

    def get_best_route_df_from_solution(self):
        self.current_solution_map = handle_weight_with_recovery(self.current_solution_map, self.config.user_model)

        _, self.new_graph = create_network_graph(self.current_solution_map)

        origin_node, dest_node, _, _, _ = self.router.set_o_d_coords(self.new_graph, self.config.gdf_coords_loaded)
        _, _, self.df_path_best = self.router.get_route(self.new_graph, origin_node, dest_node, self.heuristic_f)
        self.df_path_best = correct_arc_direction(self.df_path_best, self.data_holder.start_node_id,
                                                  self.data_holder.end_node_id)
        pass

    def calc_error(self):
        self.sub_op_list = get_virtual_op_list(self.org_df_from_io, self.current_solution_map,
                                               self.config.user_model["attrs_variable_names"])
        # demo里的要求格式
        self.sub_op_list = [(op[0], (convert(op[1][0]), to_wkt(op[1][1], rounding_precision=-1, trim=False)),
                             convert(op[2]), op[3]) for op in self.sub_op_list if op[3] == "success"]

        graph_error = len([op for op in self.sub_op_list if op[3] == "success"])

        route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                    self.config.user_model["attrs_variable_names"])

        self.data_holder.visual_detail_info['graph_error'] = graph_error
        self.data_holder.visual_detail_info['route_error'] = route_error
        self.data_holder.visual_detail_info['route_error_threshold'] = self.config.user_model["route_error_threshold"]

    def process_data_for_root_problem(self):
        # todo 把根节点当子问题 方便递归修正 未完成
        fork = self.data_holder.start_node_id
        merge = self.data_holder.end_node_id
        foil_sub_path = extract_nodes(self.df_path_foil)
        fact_sub_path = extract_nodes(self.df_path_fact)

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

    def do_solve(self):
        self.analyzer.do_basic_analyze()
        # 这里把foil的不可行变成可行 存到了current_solution_map里
        foil_must_be_feasible_arc = do_foil_must_be_feasible(self)

        counter = TrackedCounter(start=0, step=1)

        root_info = self.process_data_for_root_problem()
        root_problem = ProblemNode(self, root_info, foil_must_be_feasible_arc, self.current_solution_map,
                                   self.org_graph, None, counter, 0)
        root_problem.apply_modified_arc()
        root_problem.calc_sub_best()
        root_problem.calc_error()

        # 使用 PriorityQueue 构建优先队列
        open_queue = PriorityQueue()
        open_queue.put(root_problem)
        closed_set = set()
        # 后续可用 closed_set 记录已探索节点
        start_time = time.time()
        time_limit = 300  # 5 minutes
        while not open_queue.empty():
            if time.time() - start_time >= time_limit and self.best_leaf_node != None:
                elapsed = int(time.time() - start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                print(f"time limit reached ({minutes}分{seconds}秒) best:{self.best_leaf_node}")
                break
            problem = open_queue.get()

            closed_set.add(problem)

            if problem.route_error <= 0:
                if self.best_leaf_node is None:
                    self.best_leaf_node = problem

                    elapsed = int(time.time() - start_time)
                    minutes = elapsed // 60
                    seconds = elapsed % 60

                    print(f"first found feasible solution ({minutes}分{seconds}秒) best:{self.best_leaf_node}")

                elif problem.better_than_other(self.best_leaf_node):
                    # 找到可行解之后看看有没有更优解
                    self.best_leaf_node = problem

                continue

            self.analyzer.find_sub_forks_and_merges_node(problem.df_path_foil, problem.df_path_best,
                                                         problem.data_holder)

            info = list(problem.data_holder.foil_fact_fork_merge_nodes.values())[0]
            df_path_fact = self.generate_sub_fact(info)
            org_bc_dict = edge_betweenness_to_target_multigraph(problem.new_graph, self.data_holder.end_node_lc,
                                                                self.heuristic_f)
            modify_result_set = generate_multi_modify_arc_by_graph_feature(self, info, problem, df_path_fact,
                                                                           org_bc_dict)

            print(problem)
            for modify_arc in modify_result_set:
                # todo 这里不可能不命中，至少起点和终点是一样的
                sub_problem = ProblemNode(self, info, [modify_arc], problem.map_df, problem.new_graph, problem,
                                          problem.idx_gen, problem.level + 1)

                if sub_problem in closed_set:
                    continue

                sub_problem.apply_modified_arc()
                sub_problem.calc_sub_best()
                sub_problem.calc_error()

                if self.current_best is None or sub_problem.better_than_other(self.current_best):
                    self.current_best = sub_problem

                if self.pruning(sub_problem):
                    print(f"CUT {sub_problem}")
                    continue

                open_queue.put(sub_problem)

    def generate_sub_fact(self, info_tuple):
        nodes = info_tuple['fact_sub_path']
        fork = info_tuple['fork']
        merge = info_tuple['merge']

        path_fact = []
        for i, j in zip(nodes[:-1], nodes[1:]):
            # todo 可能数据源不应该是这里
            row = self.data_holder.get_row_info_by_arc(i, j)
            path_fact.append(row)

        df_path_fact = gpd.GeoDataFrame(path_fact, crs=self.org_map_df.crs)
        df_path_fact = correct_arc_direction(df_path_fact, fork, merge)

        return df_path_fact

    def process_solution_from_model(self):
        self.current_solution_map = self.best_leaf_node.map_df
        print(self.best_leaf_node.inherit)
        self.get_best_route_df_from_solution()
        self.calc_error()

        self.out_put_op_list = self.sub_op_list
        self.out_put_df = self.current_solution_map[self.org_df_from_io.columns]

    def pruning(self, problem) -> bool:
        if self.best_leaf_node is not None \
                and problem.not_feasible() \
                and problem.graph_error >= self.best_leaf_node.graph_error:
            # 已经找到了可行解 当前是不可行解  但是发现有graph error大于可行解的,这样是不可能找到比当前可行解好的方案
            return True

        if problem.not_feasible() \
                and problem.route_error > self.current_best.route_error \
                and problem.graph_error >= self.current_best.graph_error:
            # 当前route error 更差 但是graph error不好于当前最小
            do_pruning = self.calculate_acceptance_probability(problem) <= random.random()
            return do_pruning

        return False

    def calculate_acceptance_probability(self, problem, max_level=10):
        """
        计算接受当前解的概率，随着层数增加，不接受差解的概率增大。

        Args:
            current_best: 当前最优解
            current: 当前解
            layer: 当前层数
            level: 最大层数（用来控制层数的影响程度）

        Returns:
            probability: 接受当前解的概率
        """
        # 计算当前解与最优解的差异（以route_error或graph_error为例）
        delta = (problem.route_error - self.current_best.route_error) + (
                problem.graph_error - self.current_best.graph_error)

        if max_level == problem.level:
            acceptance_probability = 0
        else:
            acceptance_probability = math.exp((-delta) / (max_level - problem.level))

        # 将概率限制在[0, 1]范围内
        acceptance_probability = max(0, min(1, acceptance_probability))

        return acceptance_probability
