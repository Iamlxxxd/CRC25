# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 14:37
@project:    CRC25
"""
from collections import defaultdict
class MipDataHolder:
    all_nodes = []
    features = ["curb_height_max", "obstacle_free_width_float"]
    point_id_map = dict()
    all_arcs = []
    foil_route_arcs = []
    row_data = dict()

    all_feasible_arcs = defaultdict(list)
    all_infeasible_arcs = defaultdict(list)
    all_feasible_both_way = defaultdict(list)
    all_infeasible_both_way = defaultdict(list)

    all_feasible_dir_arcs = defaultdict(list)
    all_infeasible_dir_arcs = defaultdict(list)

    M:float
    visual_detail_info=dict()

    def __init__(self):
        pass
