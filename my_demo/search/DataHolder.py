# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 14:25
@project:    CRC25
"""
from collections import defaultdict
class DataHolder:
    all_nodes = []
    features = ["curb_height_max", "obstacle_free_width_float"]
    point_id_map = dict()
    all_arcs = []

    row_data = dict()

    all_feasible_arcs = defaultdict(list)
    all_infeasible_arcs = defaultdict(list)
    all_feasible_both_way = defaultdict(list)
    all_infeasible_both_way = defaultdict(list)

    all_feasible_dir_arcs = defaultdict(list)
    all_infeasible_dir_arcs = defaultdict(list)

    M:float
    visual_detail_info=dict()

    """foil"""
    foil_route_arcs = []

    foil_must_feasible_arcs = []

    """fact"""
    fact_common_from_node_arcs = dict()
    fact_common_to_node_arcs = dict()


    def __init__(self):
        pass

    def get_row_info_by_arc(self, i, j):
        return self.row_data.get((i, j), self.row_data.get((j, i), None))
