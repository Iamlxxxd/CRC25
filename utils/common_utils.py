# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/30 11:12
@project:    CRC25
"""
import os
import random
import numpy as np
import pandas as pd


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


import gurobipy as gp


def get_constraint_string(model, constr_name):
    """
    获取约束的原始表达式结构，并拼接为变量名:值的字符串形式
    返回示例: "x:1.0 + y:2.0 > z:0.5"
    """
    try:
        constr = model.getConstrByName(constr_name)
        if not constr:
            return "约束不存在"

        # 获取约束的线性表达式和运算符
        expr = model.getRow(constr)
        sense = constr.Sense  # 约束类型（'<', '>', '='）
        rhs = constr.RHS  # 右侧常量

        # 构建左侧表达式字符串
        lhs_str = ""
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            # 处理符号（正负）
            if coeff >= 0 and i > 0:
                lhs_str += " + "
            elif coeff < 0:
                lhs_str += " - "
            # 拼接变量名和值
            lhs_str += f"{var.VarName}:{var.X}"

        # 根据约束类型拼接完整表达式
        sense_symbol = {
            '<': ' < ',
            '>': ' > ',
            '=': ' == '
        }.get(sense, ' ? ')

        return f"{lhs_str}{sense_symbol}{rhs}"

    except gp.GurobiError as e:
        return f"Gurobi错误: {e}"
    except Exception as e:
        return f"未知错误: {e}"


import geopandas as gpd


def correct_arc_direction(gdf, start_node, end_node):
    # Step 1: 构建邻接字典
    from collections import defaultdict, deque
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
