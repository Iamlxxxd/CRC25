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


def generate_multi_modify_arc_by_graph_feature(solver,info, G, df_path_fact,org_bc_dict=None) -> List[tuple]:
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

    # 只遍历一次，记录最大值
    max_node_deg = -float('inf')
    max_node_deg_arc = None
    max_edge_bc = -float('inf')
    max_edge_bc_arc = None
    max_alt_ratio = -float('inf')
    max_alt_ratio_arc = None

    for idx, row in df_path_fact.iterrows():
        arc_id = row['arc']
        u, v = id_point_map.get(arc_id[0]), id_point_map.get(arc_id[1])

        # 度中心性
        node_deg_score = (deg_cent.get(u, 0) + deg_cent.get(v, 0)) / 2
        if node_deg_score > max_node_deg:
            max_node_deg = node_deg_score
            max_node_deg_arc = arc_id
        # 中介中心性
        edge_bc_score = get_edge_bc_score(u, v, G, edge_bc)
        if edge_bc_score > max_edge_bc:
            max_edge_bc = edge_bc_score
            max_edge_bc_arc = arc_id
        # 替换路径比
        alt_ratio_score = calculate_alt_ratio(u, v, G, row, weight=solver.heuristic_f)
        if alt_ratio_score > max_alt_ratio:
            max_alt_ratio = alt_ratio_score
            max_alt_ratio_arc = arc_id

    result_set = set()
    if max_node_deg_arc is not None:
        result_set.add((max_node_deg_arc, ArcModifyTag.TO_INFE))
    if max_edge_bc_arc is not None:
        result_set.add((max_edge_bc_arc, ArcModifyTag.TO_INFE))
    if max_alt_ratio_arc is not None:
        result_set.add((max_alt_ratio_arc, ArcModifyTag.TO_INFE))

    fact_id_route = info['fact_sub_path']
    fork = (fact_id_route[0],fact_id_route[1])
    merge = (fact_id_route[-2],fact_id_route[-1])

    random_node = random.choice([fork, merge])
    result_set.add((random_node, ArcModifyTag.TO_INFE))

    # todo 可以再加一个 lp解出来的候选集
    return result_set


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
