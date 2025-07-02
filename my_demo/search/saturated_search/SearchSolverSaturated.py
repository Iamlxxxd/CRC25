# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/25 10:56
@project:    CRC25
"""
from my_demo.search.SearchSolver import SearchSolver
from my_demo.config import Config
from copy import deepcopy
from typing import List
import multiprocessing
import geopandas as gpd
import pandas as pd
import networkx as nx
from networkx.classes import DiGraph
from tqdm import tqdm
from collections import defaultdict
from my_demo.config import Config
from my_demo.search.ArcModifyTag import ArcModifyTag
from my_demo.search.DataHolder import DataHolder
from my_demo.search.DataAnalyzer import DataAnalyzer
from my_demo.search.Operator import Operator
from my_demo.search.saturated_search.ProblemNode import ProblemNode
from router import Router
from utils.dataparser import create_network_graph, handle_weight, handle_weight_with_recovery
from utils.common_utils import set_seed, ensure_crs, correct_arc_direction, get_constraint_string, extract_nodes, \
    calc_bc
from utils.metrics import common_edges_similarity_route_df_weighted, get_virtual_op_list
from my_demo.visual import visual_sub_problem, visual_map_foil_modded
from my_demo.search.TrackedCounter import TrackedCounter
from queue import PriorityQueue
from my_demo.search.saturated_search.Operator import *
import time


class SearchSolverSaturated(SearchSolver):

    def __init__(self, config: Config):
        super().__init__(config)
        self.best_leaf_node: ProblemNode = None
        self.current_best: ProblemNode = None

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

        # 可以考虑计算topN
        bc_start_time = time.time()
        org_bc_dict = nx.edge_betweenness_centrality(root_problem.new_graph,k=1540)
        print("bc cost time", time.time() - bc_start_time)

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
