# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/20 10:53
@project:    CRC25
"""
from copy import deepcopy
from typing import Optional

from my_demo.search.DataHolder import DataHolder
from utils.dataparser import handle_weight_with_recovery, create_network_graph
import shapely.ops as so
import shapely.geometry as sg
import geopandas as gpd
import pandas as pd
import networkx as nx
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from utils.common_utils import correct_arc_direction
from my_demo.visual import visual_sub_problem, visual_map_foil_modded


class SubProblem:
    def __init__(self, solver, info_tuple, map_df, map_graph, master, idx_gen, level):

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
        self.map_graph = map_graph
        self.modified_row = []
        self.master = master

    def __str__(self):
        return f"{self.idx}_{self.fork}_{self.merge}"

    __repr__ = __str__

    def calc_sub_best(self):
        self.weight_df = handle_weight_with_recovery(self.map_df, self.config.user_model)

        _, self.new_graph = create_network_graph(self.weight_df)

        self.fork_lc = self.org_solver.data_holder.id_point_map.get(self.fork)
        self.merge_lc = self.org_solver.data_holder.id_point_map.get(self.merge)
        # self.origin_node, self.dest_node, self.origin_node_loc, self.dest_node_loc, _ = self.set_o_d_coords(
        #     origin_lc,
        #     dest_lc,
        #     self.org_graph)
        self.path_best, self.G_path_best, self.df_path_best = self.org_solver.router.get_route(self.new_graph,
                                                                                               self.fork_lc,
                                                                                               self.merge_lc,
                                                                                               self.org_solver.heuristic_f)
        self.df_path_best = correct_arc_direction(self.df_path_best, self.fork, self.merge)

    def process_sub_fact(self):
        nodes = self.info_tuple['fact_sub_path']

        path_fact = []
        for i, j in zip(nodes[:-1], nodes[1:]):
            # todo 可能数据源不应该是这里
            row = self.org_solver.data_holder.get_row_info_by_arc(i, j)
            path_fact.append(row)

        self.df_path_fact = gpd.GeoDataFrame(path_fact, crs=self.map_df.crs)
        self.df_path_fact = correct_arc_direction(self.df_path_fact, self.fork, self.merge)

    def process_sub_foil(self):
        nodes = self.info_tuple['foil_sub_path']

        path_foil = []
        for i, j in zip(nodes[:-1], nodes[1:]):
            # todo 可能数据源不应该是这里
            row = self.org_solver.data_holder.get_row_info_by_arc(i, j)
            path_foil.append(row)

        self.df_path_foil = gpd.GeoDataFrame(path_foil, crs=self.map_df.crs)
        self.df_path_foil = correct_arc_direction(self.df_path_foil, self.fork, self.merge)
        self.df_path_foil['mm_len'] = self.df_path_foil.geometry.length

    def set_o_d_coords(self, origin_lc, dest_lc, G):
        origin_node_loc = so.nearest_points(origin_lc, sg.MultiPoint(list(G.nodes)))[1]
        dest_node_loc = so.nearest_points(dest_lc, sg.MultiPoint(list(G.nodes)))[1]
        origin_node = (origin_node_loc.x, origin_node_loc.y)
        dest_node = (dest_node_loc.x, dest_node_loc.y)
        return origin_node, dest_node, origin_node_loc, dest_node_loc

    def process_visual_data(self):
        self.data_holder.visual_detail_info['fork'] = f"{self.fork}_{self.fork_lc}"
        self.data_holder.visual_detail_info['merge'] = f"{self.merge}_{self.merge_lc}"

        return {
            "meta_map": self.config.meta_map,
            "df_path_fact": self.df_path_fact,
            "df_path_foil": self.df_path_foil,
            "best_route": self.df_path_best,
            "org_map_df": self.map_df,
            "config": self.config,
            "data_holder": self.data_holder,
            "show_data": self.data_holder.visual_detail_info}

    def visualize_if_needed(self, tag):
        if getattr(self.config, "show_sub_problem", False):
            master_id = self.master.idx if self.master is not None else "N"
            visual_map_foil_modded(
                self.process_visual_data(),
                self.config.out_path,
                f"{master_id}|{self}##{tag}##{self.config.route_name}"
            )

    def do_solve(self):
        self.process_sub_foil()
        self.process_sub_fact()

        # todo local search  当前这个不行
        # modified_row = self.org_solver.operator_factory.make_infeasible_tail_arcs(self)
        # todo 这个减少了分支数量
        if self.idx > 0:
            # 根节点一定是失败的 而且没有识别过分叉 这里不需要操作根节点
            modified_row = self.org_solver.operator_factory.change_by_graph_feature(self)

            self.modified_row.append(modified_row)

        self.calc_sub_best()
        route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                    self.config.user_model["attrs_variable_names"])
        master_id = self.master.idx if self.master != None else "N"

        if route_error <= 0:
            # 子问题解决
            self.visualize_if_needed("SUC")
            return [self]

        while True:
            solutions = []
            self.org_solver.analyzer.find_sub_forks_and_merges_node(self.df_path_foil, self.df_path_best,
                                                                    self.data_holder)
            for fork, info in self.data_holder.foil_fact_fork_merge_nodes.items():
                # todo 这里不可能不命中，至少起点和终点是一样的
                sub_problem = SubProblem(self.org_solver, info, self.map_df, self.new_graph, self, self.idx_gen,
                                         self.level + 1)
                sub_solution = sub_problem.do_solve()
                solutions.extend(sub_solution)

            # 2. 合并子节点结果
            for s in solutions:
                for row in s.modified_row:
                    self.modified_row.append(row)
                    self.map_df.loc[row.name] = row

            # 3. 重新计算自身
            self.calc_sub_best()
            route_error = 1 - common_edges_similarity_route_df_weighted(self.df_path_best, self.df_path_foil,
                                                                        self.config.user_model["attrs_variable_names"])

            if route_error <= 0:
                tag = "R#SUC"
                if self.idx == 0:
                    tag = "ROOT#SUC#"
                # 子问题解决
                self.visualize_if_needed(tag)
                return [self]

            # 子问题未解决
            self.visualize_if_needed("FAIL")
            if self.idx_gen.peek() > 50 or self.level > 20:
                # 探索节点过多 或者深度过多
                break
        return solutions
