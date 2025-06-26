# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 16:22
@project:    CRC25
"""
from my_demo.search.ArcModifyTag import ArcModifyTag
import networkx as nx
import math
from collections import defaultdict


class Operator:
    def __init__(self, solver):
        self.solver = solver
        self.data_holder = self.solver.data_holder

    def do_foil_must_be_feasible(self):
        for i, j in self.data_holder.foil_must_feasible_arcs:
            modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_FE)
            solution_row = self.solver.current_solution_map.loc[modified_row.name]
            modified_row['modified'] = modified_row['modified'] + solution_row['modified']
            self.solver.current_solution_map.loc[modified_row.name] = modified_row

    def do_must_be_infeasible_arcs(self):
        """公共节点 但是分叉 按说要断掉？ 但是好像只需要断一侧"""
        for from_node, arc_list in self.data_holder.fact_common_from_node_arcs.items():
            for i, j in arc_list:
                modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
                solution_row = self.solver.current_solution_map.loc[modified_row.name]
                modified_row['modified'] = modified_row['modified'] + solution_row['modified']
                self.solver.current_solution_map.loc[modified_row.name] = modified_row

        for to_node, arc_list in self.data_holder.fact_common_to_node_arcs.items():
            for i, j in arc_list:
                modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
                solution_row = self.solver.current_solution_map.loc[modified_row.name]
                modified_row['modified'] = modified_row['modified'] + solution_row['modified']
                self.solver.current_solution_map.loc[modified_row.name] = modified_row

    def make_infeasible_tail_arcs(self, problem):
        i, j = problem.sub_fact[-2], problem.sub_fact[-1]
        modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
        solution_row = problem.map_df.loc[modified_row.name]
        modified_row['modified'] = modified_row['modified'] + solution_row['modified']
        problem.map_df.loc[modified_row.name] = modified_row

        return modified_row

    def change_by_graph_feature(self, problem):
        results = self.evaluate_fact_edges(problem)
        i,j = results[0]['arc_id']

        modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
        solution_row = problem.map_df.loc[modified_row.name]
        modified_row['modified'] = modified_row['modified'] + solution_row['modified']
        problem.map_df.loc[modified_row.name] = modified_row

        return modified_row

    def evaluate_fact_edges(self, problem):
        """
        评估fact路径上各边的断开优先级，先分别归一化4个指标，再加权累加

        返回:
        包含每条边详细评分和排名的字典
        """
        G = problem.map_graph

        # 1. 计算所有指标
        deg_cent = nx.degree_centrality(G)
        edge_bc = nx.edge_betweenness_centrality(G)

        def calculate_alt_ratio(u, v, row, weight='dijkstra'):
            try:
                orig_weight = row['my_weight']
                edge_data = G.get_edge_data(u, v)
                G.remove_edge(u, v)
                try:
                    alt_length = nx.shortest_path_length(G, u, v, weight=weight)
                    ratio = alt_length / orig_weight
                except nx.NetworkXNoPath:
                    ratio = float('inf')
                finally:
                    G.add_edge(u, v, **edge_data)
                return ratio
            except KeyError:
                return float('inf')

        def get_edge_bc_score(u, v, G, edge_bc):
            for edge_id in G.get_edge_data(u, v).keys():
                if (u, v, edge_id) in edge_bc:
                    return edge_bc[(u, v, edge_id)]
            return 0

        # 2. 统计所有边的原始指标
        edge_metrics = []
        for idx, row in problem.df_path_fact.iterrows():
            arc_id = row['arc']
            u, v = self.data_holder.id_point_map.get(arc_id[0]), self.data_holder.id_point_map.get(arc_id[1])
            #度中心性
            node_deg_score = (deg_cent.get(u, 0) + deg_cent.get(v, 0)) / 2
            #中介中心性
            edge_bc_score = get_edge_bc_score(u, v, G, edge_bc)
            #替换路径比
            alt_ratio_score = calculate_alt_ratio(u, v, row, weight=problem.org_solver.heuristic_f)
            # alt_ratio_score = math.log10(alt_ratio + 1) if alt_ratio < float('inf') else 2.0
            position_score = 1.0
            if u == problem.fork or v == problem.fork or u == problem.merge or v == problem.merge:
                position_score = 1.5
            edge_metrics.append({
                'arc_id': arc_id,
                'arc': (u, v),
                'node_deg_score': node_deg_score,
                'edge_bc_score': edge_bc_score,
                'alt_ratio_score': alt_ratio_score,
                'position_score': position_score
            })

        # 3. 分别归一化四个指标
        def normalize(arr):
            min_v, max_v = min(arr), max(arr)
            if max_v == min_v:
                return [0.0 for _ in arr]
            return [(x - min_v) / (max_v - min_v) for x in arr]

        node_deg_list = [x['node_deg_score'] for x in edge_metrics]
        edge_bc_list = [x['edge_bc_score'] for x in edge_metrics]
        alt_ratio_list = [x['alt_ratio_score'] for x in edge_metrics]
        position_list = [x['position_score'] for x in edge_metrics]

        node_deg_norm = normalize(node_deg_list)
        edge_bc_norm = normalize(edge_bc_list)
        alt_ratio_norm = normalize(alt_ratio_list)
        position_norm = normalize(position_list)

        # 4. 加权累加
        weights = {
            'alt_ratio': 0.45,
            'edge_bc': 0.35,
            'node_deg': 0.15,
            'position': 0.05
        }
        results = {}
        for i, metric in enumerate(edge_metrics):
            composite_score = (
                alt_ratio_norm[i] * weights['alt_ratio'] +
                edge_bc_norm[i] * weights['edge_bc'] +
                node_deg_norm[i] * weights['node_deg'] +
                position_norm[i] * weights['position']
            )
            arc_id = metric['arc_id']
            results[arc_id] = {
                'node_deg_score': metric['node_deg_score'],
                'edge_bc_score': metric['edge_bc_score'],
                'alt_ratio_score': metric['alt_ratio_score'],
                'position_score': metric['position_score'],
                'node_deg_norm': node_deg_norm[i],
                'edge_bc_norm': edge_bc_norm[i],
                'alt_ratio_norm': alt_ratio_norm[i],
                'position_norm': position_norm[i],
                'composite_score': composite_score
            }

        sorted_edges = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        result_list = []
        for rank, (arc_id, data) in enumerate(sorted_edges, 1):
            data['rank'] = rank
            data['arc_id'] = arc_id
            result_list.append(data)

        return result_list
