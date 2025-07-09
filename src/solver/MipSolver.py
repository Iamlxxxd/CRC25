#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/29 17:43
# @Author  : JunhaoShi(01387247)
# @Desc    :

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from scipy.optimize import linprog
from scipy.sparse import dok_matrix
from shapely import to_wkt

from my_demo.config import Config
from src.DataHolder import DataHolder
from src.calc.common_utils import correct_arc_direction
from src.calc.dataparser import convert
from src.calc.dataparser import handle_weight, handle_weight_with_recovery, create_network_graph
from src.calc.metrics import get_virtual_op_list, common_edges_similarity_route_df_weighted
from src.calc.router import Router
from src.solver.ArcModifyTag import ArcModifyTag


class MipSolver:
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

        self.mip_modify_arc_list = []
        self.big_m = self.data_holder.M

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

                        self.data_holder.all_feasible_arcs[ez_attr].append((node1, node2))
                        if node1 != node2:
                            self.data_holder.all_feasible_arcs[ez_attr].append((node2, node1))
                            self.data_holder.all_feasible_both_way[ez_attr].append((node1, node2))

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

        df['curb_height_max'].fillna(0, inplace=True)
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

    def process_solution_from_model(self, exclude_arcs=None):
        if exclude_arcs is None:
            exclude_arcs = []
        self.save_modify_arc_from_mip(exclude_arcs)
        self.apply_mip_modified_arc()

        self.get_best_route_df_from_solution()
        self.data_holder.final_weight = self.current_solution_map.set_index("arc").to_dict()['my_weight']
        self.fill_w_value_for_visual()

        self.calc_error()
        self.out_put_op_list = self.sub_op_list
        self.out_put_df = self.current_solution_map[self.org_df_from_io.columns]

    def get_best_route_df_from_solution(self):
        self.current_solution_map = handle_weight_with_recovery(self.current_solution_map, self.config.user_model)

        _, best_graph = create_network_graph(self.current_solution_map)

        origin_node, dest_node, _, _, _ = self.router.set_o_d_coords(best_graph, self.config.gdf_coords_loaded)
        _, _, self.df_path_best = self.router.get_route(best_graph, origin_node, dest_node, self.heuristic_f)
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

    def save_modify_arc_from_mip(self, exclude_arcs=None):
        if exclude_arcs is None:
            exclude_arcs = []

        feature_modify_mark = set()
        for i, j_dict in self.y.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
                if (i, j) in feature_modify_mark or (j, i) in feature_modify_mark:
                    continue

                row = self.get_row_info_by_arc(i, j)
                if row is None:
                    continue
                if value >= 0.99:
                    if row['include'] < 1:
                        # 原来不可行 但是改成可行
                        self.mip_modify_arc_list.append(((i, j), ArcModifyTag.TO_FE))
                else:
                    if row['include'] > 0:
                        # 原来可行 但是改成不可行
                        self.mip_modify_arc_list.append(((i, j), ArcModifyTag.TO_INFE))
                feature_modify_mark.add((i, j))

        type_modify_mark = set()
        for i, j_dict in self.x.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
                if (i, j) in type_modify_mark or (j, i) in type_modify_mark:
                    continue

                row = self.get_row_info_by_arc(i, j)
                if row is None:
                    continue

                if value >= 0.99:
                    self.mip_modify_arc_list.append(((i, j), ArcModifyTag.CHANGE))
                    type_modify_mark.add((i, j))

    def apply_mip_modified_arc(self):
        for (i, j), modify_tag in self.mip_modify_arc_list:
            modified_row = self.modify_df_arc_with_attr(i, j, modify_tag)
            solution_row = self.current_solution_map.loc[modified_row.name]
            modified_row['modified'] = modified_row['modified'] + solution_row['modified']
            self.current_solution_map.loc[modified_row.name] = modified_row

    def modify_df_arc_with_attr(self, i, j, tag, attr_name=None):
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

    def fill_w_value_for_visual(self):
        self.current_solution_map['w_i'] = 0
        self.current_solution_map['w_j'] = 0
        self.current_solution_map['y_ij'] = 0

        for idx, row in self.current_solution_map.iterrows():
            arc = row['arc']
            self.current_solution_map.at[row.name, "w_i"] = self.w[arc[0]]
            self.current_solution_map.at[row.name, "w_j"] = self.w[arc[1]]
            self.current_solution_map.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]

        self.df_path_foil['w_i'] = 0
        self.df_path_foil['w_j'] = 0
        self.df_path_foil['y_ij'] = 0

        cost = 0
        for idx, row in self.df_path_foil.iterrows():
            arc = row['arc']
            self.df_path_foil.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_path_foil.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_path_foil.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["foil_cost"] = round(cost, 4)

        self.df_path_best['w_i'] = 0
        self.df_path_best['w_j'] = 0
        self.df_path_best['y_ij'] = 0
        cost = 0
        for idx, row in self.df_path_best.iterrows():
            arc = row['arc']
            self.df_path_best.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_path_best.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_path_best.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["best_cost"] = round(cost, 4)

        self.df_path_fact['w_i'] = 0
        self.df_path_fact['w_j'] = 0
        self.df_path_fact['y_ij'] = 0
        cost = 0
        for idx, row in self.df_path_fact.iterrows():
            arc = row['arc']
            self.df_path_fact.at[row.name, "w_i"] = self.w[arc[0]]
            self.df_path_fact.at[row.name, "w_j"] = self.w[arc[1]]
            self.df_path_fact.at[row.name, "y_ij"] = self.y[arc[0]][arc[1]]
            cost += self.data_holder.final_weight.get(arc, self.data_holder.final_weight.get((arc[1], arc[0]),
                                                                                             -self.data_holder.M))
        self.data_holder.visual_detail_info["fact_cost"] = round(cost, 4)
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
                "data_holder": self.data_holder,
                "show_data": self.data_holder.visual_detail_info}

    def init_model(self):
        # ========== 1. 变量索引映射 ==========
        nodes = sorted(self.data_holder.all_nodes)
        arcs = sorted(self.data_holder.all_arcs)  # 所有有向弧
        # 是否整数变量
        var_type = []
        # 创建索引映射
        var_index = {}
        idx = 0

        # w_i 变量
        w_index = {node: idx for idx, node in enumerate(nodes)}
        idx += len(nodes)
        var_type += [0] * len(nodes)

        # x_ij 变量
        x_index = {}
        for arc in arcs:
            var_index[('x', arc)] = idx
            x_index[arc] = idx
            idx += 1
        var_type += [1] * len(arcs)

        # y_ij 变量
        y_index = {}
        for arc in arcs:
            var_index[('y', arc)] = idx
            y_index[arc] = idx
            idx += 1
        var_type += [1] * len(arcs)

        total_vars = idx
        self.var_index = var_index
        self.w_index = w_index
        self.x_index = x_index
        self.y_index = y_index

        # ========== 2. 构建目标向量 c ==========
        c = np.zeros(total_vars)

        for arc in arcs:
            row = self.get_row_info_by_arc(*arc)
            # x_ij 的系数
            c[x_index[arc]] = 1
            # y_ij 的系数
            c[y_index[arc]] = row['Deleta_p'] - row['Deleta_n']

        # ========== 3. 边界条件 ==========
        bounds = []
        # w_i >= 0
        bounds.extend([(0, None)] * len(nodes))
        # 0 <= x_ij <= 1
        bounds.extend([(0, 1)] * len(arcs))
        # 0 <= y_ij <= 1
        bounds.extend([(0, 1)] * len(arcs))

        # ========== 4. 约束构建 ==========
        # A_ub = []  # 不等式约束矩阵
        A_ub = dok_matrix((len(arcs) + len(self.data_holder.foil_route_arcs), total_vars))
        b_ub = []  # 不等式右侧向量
        # A_eq = []  # 等式约束矩阵

        num_fe_symmetry = sum(1 for group in self.data_holder.all_feasible_both_way.values() for (i, j) in group)
        num_in_symmetry = sum(1 for group in self.data_holder.all_infeasible_both_way.values() for (i, j) in group)
        num_unchange = sum(
            1 for (i, j) in self.data_holder.all_arcs if not self.get_row_info_by_arc(i, j)['modify_able_path_type'])
        total_constrs = len(self.data_holder.foil_route_arcs) + 2 * num_fe_symmetry + 2 * num_in_symmetry + num_unchange
        A_eq = dok_matrix((total_constrs, total_vars))
        b_eq = []  # 等式右侧向量

        # (1) Shortest Path Constraints
        A_ub_idx = 0
        A_eq_idx = 0
        for (i, j) in arcs:
            row = self.get_row_info_by_arc(i, j)
            c_ij = row['c']
            d_ij = row['d']

            # w_j - w_i 项
            A_ub[A_ub_idx, w_index[j]] = 1
            A_ub[A_ub_idx, w_index[i]] = -1
            # -d_ij * x_ij 项
            A_ub[A_ub_idx, x_index[(i, j)]] = -d_ij
            # -big_m * y_ij 项
            A_ub[A_ub_idx, y_index[(i, j)]] = self.big_m

            b_ub.append(c_ij + self.big_m)
            A_ub_idx += 1

        # foil route Shortest Path Constraints
        for (i, j) in self.data_holder.foil_route_arcs:
            row = self.get_row_info_by_arc(i, j)
            c_ij = row['c']
            d_ij = row['d']

            # foil route feasible
            A_eq[A_eq_idx, y_index[(i, j)]] = 1
            b_eq.append(1)

            # foil route Shortest path
            # w_i - w_j 项
            A_ub[A_ub_idx, w_index[i]] = 1
            A_ub[A_ub_idx, w_index[j]] = -1
            # d_ij * x_ij 项
            A_ub[A_ub_idx, x_index[(i, j)]] = d_ij
            # big_m * y_ij 项
            A_ub[A_ub_idx, y_index[(i, j)]] = self.big_m

            b_ub.append(self.big_m - c_ij)
            A_ub_idx += 1
            A_eq_idx += 1

        # (3) Undirected Arc Constraint
        for f, arcs in self.data_holder.all_feasible_both_way.items():
            for i, j in arcs:
                # x_ij = x_ji
                A_eq[A_eq_idx, x_index[(i, j)]] = 1
                A_eq[A_eq_idx, x_index[(j, i)]] = -1
                b_eq.append(0)
                A_eq_idx += 1

                # y_ij = y_ji
                A_eq[A_eq_idx, y_index[(i, j)]] = 1
                A_eq[A_eq_idx, y_index[(j, i)]] = -1
                b_eq.append(0)
                A_eq_idx += 1
        for f, arcs in self.data_holder.all_infeasible_both_way.items():
            for i, j in arcs:
                # x_ij = x_ji
                A_eq[A_eq_idx, x_index[(i, j)]] = 1
                A_eq[A_eq_idx, x_index[(j, i)]] = -1
                b_eq.append(0)
                A_eq_idx += 1

                # y_ij = y_ji
                A_eq[A_eq_idx, y_index[(i, j)]] = 1
                A_eq[A_eq_idx, y_index[(j, i)]] = -1
                b_eq.append(0)
                A_eq_idx += 1

        # (4) can not modify path_type
        for (i, j) in self.data_holder.all_arcs:
            row = self.get_row_info_by_arc(i, j)
            if not row['modify_able_path_type']:
                A_eq[A_eq_idx, x_index[(i, j)]] = 1
                b_eq.append(0)
                A_eq_idx += 1

        # ========== 5. 保存约束矩阵 ==========
        # self.A_ub = np.array(A_ub) if A_ub else None
        self.A_ub = A_ub
        self.b_ub = np.array(b_ub) if b_ub else None
        self.A_eq = A_eq
        self.b_eq = np.array(b_eq) if b_eq else None
        self.c = c
        self.bounds = bounds
        self.var_type = var_type
        print(self.A_ub.shape)

    def solve_model(self, time_limit=3600, gap=0):
        opt = {'disp': True,
               'mip_rel_gap': gap,
               'time_limit': time_limit,
               # 'log_file': str(os.path.join(self.config.base_dir, "my_demo", "output", "solver_log.txt"))
               }
        # 注意: scipy 1.10 不支持时间限制和间隙控制
        result = linprog(
            c=self.c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method='highs',  # HiGHS 求解器
            integrality=self.var_type,
            options=opt
        )

        if not result.success:
            raise RuntimeError(f"求解失败: {result.message}")

        # 将解保存到变量字典
        self.solution = result.x
        self._save_solution_to_vars()

    def _save_solution_to_vars(self):
        """将解向量赋值回字典变量"""
        nodes = sorted(self.data_holder.all_nodes)
        arcs = self.data_holder.all_arcs

        # 初始化字典
        self.w = {node: 0.0 for node in nodes}
        self.x = defaultdict(dict)
        self.y = defaultdict(dict)

        # 填充w
        for node in nodes:
            self.w[node] = self.solution[self.w_index[node]]

        # 填充x和y
        for arc in arcs:
            i, j = arc
            x_val = self.solution[self.x_index[arc]]
            y_val = self.solution[self.y_index[arc]]

            self.x[i][j] = x_val
            self.y[i][j] = y_val

            # 确保对称性
            if (j, i) in arcs:
                self.x[j][i] = x_val
                self.y[j][i] = y_val
