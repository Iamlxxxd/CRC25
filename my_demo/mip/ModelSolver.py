# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 10:35
@project:    CRC25
"""
from copy import deepcopy
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from my_demo.config import Config
from my_demo.mip.MipDataHolder import MipDataHolder
from router import Router
from utils.dataparser import create_network_graph, handle_weight, handle_weight_with_recovery
from utils.common_utils import correct_arc_direction


class ModelSolver:
    heuristic_f = 'my_weight'
    heuristic = "dijkstra"

    eazy_name_map = {"curb_height_max": "H", "obstacle_free_width_float": "W"}
    eazy_name_map_reversed = {"H": "curb_height_max", "W": "obstacle_free_width_float"}

    def __init__(self, config: Config):
        self.config = config
        self.meta_map = config.meta_map
        self.data_holder = MipDataHolder()

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
        self.org_map_df['org_obstacle_free_width_float'] = self.org_map_df['obstacle_free_width_float']
        self.org_map_df['org_curb_height_max'] = self.org_map_df['curb_height_max']
        self.handle_weight_without_preference(self.org_map_df)

        self.process_nodes()
        self.process_arcs()
        self.process_foil()
        self.process_fact()
        pass

    def process_nodes(self):
        all_points = pd.concat([self.org_map_df['geometry'].apply(lambda x: x.coords[0]),
                                self.org_map_df['geometry'].apply(lambda x: x.coords[1])]).drop_duplicates().tolist()
        point_id_map = {pt: idx for idx, pt in enumerate(all_points)}
        self.data_holder.all_nodes = list(point_id_map.values())
        self.data_holder.point_id_map = point_id_map

        self.data_holder.start_node = self.data_holder.point_id_map.get(self.origin_node)
        self.data_holder.end_node = self.data_holder.point_id_map.get(self.dest_node)

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

        df['curb_height_max'].fillna(0,inplace=True)
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

        # if user_model["walk_bike_preference"] == 'walk':
        #     df.loc[df['path_type'] == 'walk', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        # elif user_model["walk_bike_preference"] == 'bike':
        #     df.loc[df['path_type'] == 'bike', 'my_weight'] = df['my_weight'] * user_model["walk_bike_preference_weight_factor"]
        #
        # df['my_weight'] = df['my_weight'] /df['my_weight'].abs().max()

    def get_row_info_by_arc(self, i, j):
        return self.data_holder.row_data.get((i, j), self.data_holder.row_data.get((j, i), None))

    def process_solution_from_model(self, exclude_arcs=None):
        if exclude_arcs is None:
            exclude_arcs = []
        self.modify_org_map_df_by_solution(exclude_arcs)
        self.get_best_route_df_from_solution()
        self.data_holder.final_weight = self.org_map_df.set_index("arc").to_dict()['my_weight']
        self.fill_w_value_for_visual()

    def modify_org_map_df_by_solution(self, exclude_arcs=None):
        if exclude_arcs is None:
            exclude_arcs = []
        # todo unfinished
        feature_modify_mark = set()
        self.graph_error = 0
        for i, j_dict in self.y.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
                if (i, j) in feature_modify_mark or (j, i) in feature_modify_mark:
                    continue
                # if value.varValue >0.99:
                if value.varValue == 1:
                    self.modify_df_arc_with_attr(i, j, "to_fe")
                else:
                    self.modify_df_arc_with_attr(i, j, "to_infe")
                feature_modify_mark.add((i, j))

        type_modify_mark = set()
        for i, j_dict in self.x.items():
            for j, value in j_dict.items():
                if (i, j) in exclude_arcs or (j, i) in exclude_arcs:
                    continue
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

    def get_best_route_df_from_solution(self):
        self.org_map_df = handle_weight_with_recovery(self.org_map_df, self.config.user_model)

        _, best_graph = create_network_graph(self.org_map_df)

        origin_node, dest_node, _, _, _ = self.router.set_o_d_coords(best_graph, self.config.gdf_coords_loaded)
        _, _, self.df_best_route = self.router.get_route(best_graph, origin_node, dest_node, self.heuristic_f)
        self.df_best_route = correct_arc_direction(self.df_best_route, self.data_holder.start_node,
                                                   self.data_holder.end_node)
        pass

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

    def print_infeasible(self, model):
        model.computeIIS()
        infeasible_list = []
        for cons in model.getConstrs():
            if cons.IISConstr:
                print(cons.constrName)
                infeasible_list.append(cons.constrName)
        model.write('model.ilp')
        return infeasible_list
