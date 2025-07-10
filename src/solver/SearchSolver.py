# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/25 10:56
@project:    CRC25
"""
from copy import deepcopy
import geopandas as gpd
from queue import PriorityQueue
import time
import random
import math

from config import Config
from src.calc.DataAnalyzer import DataAnalyzer
from src.solver.BaseSolver import BaseSolver
from src.solver.ProblemNode import ProblemNode
from src.utils.common_utils import correct_arc_direction, extract_nodes, edge_betweenness_to_target_multigraph, \
    time_over_check
from src.TrackedCounter import TrackedCounter
from src.solver.Operator import do_foil_must_be_feasible, generate_multi_modify_arc_by_graph_feature


class SearchSolver(BaseSolver):
    def __init__(self, config: Config, start_time):
        super().__init__(config, start_time)

        self.best_leaf_node: ProblemNode = None
        self.current_best: ProblemNode = None

    def init_from_config(self):
        super().init_from_config()
        self.analyzer = DataAnalyzer(self)

    def init_from_other_solver(self, other_solver: BaseSolver):
        self.data_holder = other_solver.data_holder
        self.data_holder.visual_detail_info = dict()

        self.router = other_solver.router
        self.org_map_df = other_solver.org_map_df
        self.current_solution_map = deepcopy(self.org_map_df)
        self.org_df_from_io = other_solver.org_df_from_io
        self.org_graph = other_solver.org_graph
        self.origin_node = other_solver.origin_node
        self.dest_node = other_solver.dest_node
        self.origin_node_loc = other_solver.origin_node_loc
        self.dest_node_loc = other_solver.dest_node_loc
        self.path_fact = other_solver.path_fact
        self.G_path_fact = other_solver.G_path_fact
        self.df_path_fact = other_solver.df_path_fact
        self.df_path_foil = other_solver.df_path_foil

        for (i, j), modify_type in other_solver.modify_arc_solution:
            self.modify_arc_dict[(i, j)] = modify_type

        self.analyzer = DataAnalyzer(self)

    def process_solution_from_model(self):
        self.current_solution_map = self.best_leaf_node.map_df
        print(self.best_leaf_node.inherit)
        self.modify_arc_solution = self.best_leaf_node.inherit
        self.get_best_route_df_from_solution()
        self.calc_error()

        self.out_put_op_list = self.sub_op_list
        self.out_put_df = self.current_solution_map[self.org_df_from_io.columns]

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
        while not open_queue.empty():
            if time_over_check(self.start_time, self.time_limit):
                elapsed = int(time.time() - self.start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                print(f"time limit reached ({minutes}分{seconds}秒) best:{self.best_leaf_node}")
                break

            problem = open_queue.get()

            closed_set.add(problem)

            if problem.route_error <= 0:
                if self.best_leaf_node is None:
                    self.best_leaf_node = problem

                    elapsed = int(time.time() - self.start_time)
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

        if self.best_leaf_node is None:
            if self.current_best is None:
                self.best_leaf_node = root_problem
            else:
                # 如果没找到可行解  就返回当前最好的解
                self.best_leaf_node = self.current_best

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
