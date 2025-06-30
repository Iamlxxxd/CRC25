# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/25 11:41
@project:    CRC25
"""
from operator import itemgetter

from my_demo.search.ArcModifyTag import ArcModifyTag
import networkx as nx
import math
from collections import defaultdict
from typing import List
import random


def do_foil_must_be_feasible(root_solver) -> List[tuple]:
    modified_arc_list = []
    for arc in root_solver.data_holder.foil_must_feasible_arcs:
        modified_arc_list.append((arc, ArcModifyTag.TO_FE))

    return modified_arc_list


def generate_multi_modify_arc_by_graph_feature(solver, info, G, df_path_fact, org_bc_dict=None) -> List[tuple]:
    """
    Args:
        father_problem:
        info_tuple: 第一个分叉再交汇的子结构
    Returns:
    """
    id_point_map = solver.data_holder.id_point_map

    deg_cent = nx.degree_centrality(G)
    if org_bc_dict is None:
        edge_bc = nx.edge_betweenness_centrality(G)
    else:
        edge_bc = org_bc_dict

    arc_feature_list = []
    fact_id_route = info['fact_sub_path']
    fork = (fact_id_route[0], fact_id_route[1])
    merge = (fact_id_route[-2], fact_id_route[-1])
    random_arc = random.choice([fork, merge])

    # 先收集所有特征
    for idx, row in df_path_fact.iterrows():
        arc_id = row['arc']
        u, v = id_point_map.get(arc_id[0]), id_point_map.get(arc_id[1])

        # 特征1：度中心性
        node_deg_score = (deg_cent.get(u, 0) + deg_cent.get(v, 0)) / 2
        # 特征2：中介中心性
        edge_bc_score = get_edge_bc_score(u, v, G, edge_bc)
        # 特征3：替换路径比
        alt_ratio_score = calculate_alt_ratio(u, v, G, row, weight=solver.heuristic_f)
        # 特征4：是否为随机arc
        rand_score = 1.5 if arc_id == random_arc else 1
        arc_feature_list.append({
            'arc_id': arc_id,
            'node_deg': node_deg_score,
            'edge_bc': edge_bc_score,
            'alt_ratio': alt_ratio_score,
            'fork_merge': rand_score
        })

    # 归一化（去除reverse参数和翻转逻辑）
    def normalize(feat_list, key):
        vals = [x[key] for x in feat_list]
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [0.0 for _ in vals]
        return [(v - min_v) / (max_v - min_v) for v in vals]

    norm_deg = normalize(arc_feature_list, 'node_deg')
    norm_bc = normalize(arc_feature_list, 'edge_bc')
    norm_alt = normalize(arc_feature_list, 'alt_ratio')
    norm_rand = normalize(arc_feature_list, 'fork_merge')

    arc_scores = []
    for i, feat in enumerate(arc_feature_list):
        score = (
                0.15 * norm_deg[i] +
                0.35 * norm_bc[i] +
                0.45 * norm_alt[i] +
                0.05 * norm_rand[i]
        )
        arc_scores.append((feat['arc_id'], score))

    # 按分数降序排序，取topN
    arc_scores.sort(key=lambda x: x[1], reverse=True)
    result = []
    for arc_id, _ in arc_scores[:4]:
        result.append((arc_id, ArcModifyTag.TO_INFE))

    return result


def calculate_alt_ratio(u, v, G, row, weight='dijkstra'):
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
