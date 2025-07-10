# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/30 11:12
@project:    CRC25
"""
import heapq
import math
import os
import random
from collections import defaultdict
from operator import truediv

import geopandas as gpd

import numpy as np


def set_seed(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)


def ensure_crs(gdf, crs):
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs != crs:
        gdf = gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def correct_arc_direction(gdf, start_node, end_node):
    # Step 1: 构建邻接字典
    from collections import defaultdict
    neighbors = defaultdict(list)
    edge_map = {}

    for idx, (u, v) in gdf['arc'].items():
        neighbors[u].append(v)
        neighbors[v].append(u)
        edge_map[frozenset((u, v))] = idx  # 无向边 key -> DataFrame 索引

    # Step 2: 路径追踪，从起点走到终点
    path = [start_node]
    visited = set()
    current = start_node

    while current != end_node:
        visited.add(current)
        next_nodes = neighbors[current]
        found_next = False
        for node in next_nodes:
            if node not in visited:
                path.append(node)
                current = node
                found_next = True
                break
        if not found_next:
            raise ValueError("无法从起点走到终点，请确认路径连通且无环")

    # Step 3: 重建有向 arc 顺序
    new_rows = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        idx = edge_map[frozenset((u, v))]
        row = gdf.loc[idx].copy()
        row['arc'] = (u, v)  # 修正方向
        new_rows.append(row)

    # Step 4: 返回新的 GeoDataFrame（只保留路径部分）
    return gpd.GeoDataFrame(new_rows, columns=gdf.columns, crs=gdf.crs)


def extract_nodes(df):
    nodes = []
    for idx, row in df.iterrows():
        i, j = row['arc']
        if not nodes:
            nodes.append(i)
        nodes.append(j)
    return nodes


def edge_betweenness_to_target_multigraph(G, target, weight=None):
    # 初始化数据结构
    dist = defaultdict(lambda: float('inf'))
    sigma = defaultdict(int)
    pred = defaultdict(list)  # 存储(前驱节点, 边key)
    delta = defaultdict(float)

    # 初始化结果字典 - 包含所有边，初始值为0
    betweenness = {}
    for u, v, key in G.edges(keys=True):
        betweenness[(u, v, key)] = 0.0

    # 目标节点初始化
    dist[target] = 0
    sigma[target] = 1

    # 优先队列 (距离, 节点)
    heap = [(0, target)]

    # 构建反向图邻接表（精确匹配NetworkX的边顺序）
    rev_adj = defaultdict(lambda: defaultdict(list))
    for u, v, key, data in G.edges(keys=True, data=True):
        rev_adj[v][u].append((key, data))

    # 确保边处理顺序一致（按key排序）
    for u in rev_adj:
        for v in rev_adj[u]:
            rev_adj[u][v] = sorted(rev_adj[u][v], key=lambda x: x[0])

    # Dijkstra算法 (反向图) - 精确匹配NetworkX
    while heap:
        d, w = heapq.heappop(heap)
        if d != dist[w]:
            continue

        # 处理反向图中的出边 (即原图的入边)
        if w in rev_adj:
            for u, edges in rev_adj[w].items():
                for key, data in edges:
                    # 获取边权重（精确匹配NetworkX的权重处理）
                    edge_weight = data.get(weight, 1) if weight else 1
                    if edge_weight <= 0:
                        edge_weight = 1  # 处理负权，与NetworkX一致
                    new_dist = dist[w] + edge_weight

                    # 使用更严格的浮点比较
                    if new_dist < dist[u] - 1e-12:
                        dist[u] = new_dist
                        sigma[u] = sigma[w]
                        pred[u] = [(w, key)]  # 存储前驱节点和边key
                        heapq.heappush(heap, (new_dist, u))

                    # 找到等长路径（使用相对容差比较）
                    elif math.isclose(new_dist, dist[u], rel_tol=1e-9, abs_tol=1e-12):
                        sigma[u] += sigma[w]
                        pred[u].append((w, key))

    # 按距离排序节点 (从大到小) - 匹配NetworkX的排序顺序
    nodes_by_dist = sorted(dist.keys(), key=lambda x: (dist[x], str(x)), reverse=True)

    # 反向传播计算边贡献 - 精确匹配累积顺序
    for w in nodes_by_dist:
        if w == target:
            continue

        # 对前驱节点排序以匹配NetworkX处理顺序
        sorted_pred = sorted(pred[w], key=lambda x: (x[0], x[1]))

        for pred_info in sorted_pred:
            v, key = pred_info
            # 使用更高精度计算
            ratio = sigma[v] / sigma[w] if sigma[w] > 0 else 0.0
            edge_contribution = ratio * (1.0 + delta[w])

            # 更新边中心性 (原图方向)
            edge_id = (w, v, key)
            if edge_id in betweenness:
                betweenness[edge_id] += edge_contribution

            # 更新前驱节点的delta值（使用精确累加）
            delta[v] += edge_contribution

    # 使用NetworkX相同的归一化方法和顺序
    n = G.number_of_nodes()
    if n <= 1:
        scale = None
    else:
        # 与NetworkX一致：1/(n(n-1)) for directed graphs
        scale = 1.0 / (n * (n - 1))

    if scale is not None:
        # 应用归一化（使用更高精度乘法）
        for edge_id in betweenness:
            betweenness[edge_id] = betweenness[edge_id] * scale

    return betweenness


def compare_dicts(dict1, dict2):
    """
    递归比较两个字典的差异，返回：
    - only_in_dict1: 仅在dict1存在的键路径
    - only_in_dict2: 仅在dict2存在的键路径
    - value_diff: 相同键但值不同的路径及对应值
    """
    diff = {"only_in_dict1": [], "only_in_dict2": [], "value_diff": []}

    def _compare(d1, d2, path=""):
        # 检查dict1独有的键
        for k in set(d1.keys()) - set(d2.keys()):
            new_path = f"{path}.{k}" if path else k
            diff["only_in_dict1"].append(new_path)

        # 检查dict2独有的键
        for k in set(d2.keys()) - set(d1.keys()):
            new_path = f"{path}.{k}" if path else k
            diff["only_in_dict2"].append(new_path)

        # 比较共有键
        for k in set(d1.keys()) & set(d2.keys()):
            new_path = f"{path}.{k}" if path else k
            v1, v2 = d1[k], d2[k]

            # 嵌套字典递归比较
            if isinstance(v1, dict) and isinstance(v2, dict):
                _compare(v1, v2, new_path)
            # 值直接比较
            elif v1 != v2:
                diff["value_diff"].append({
                    "path": new_path,
                    "dict1_value": v1,
                    "dict2_value": v2
                })

    _compare(dict1, dict2)
    return diff


import time


def time_over_check(start_time, time_limit) -> bool:
    if time.time() - start_time >= time_limit:
        return True

    return False
