# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 14:25
@project:    CRC25
"""
from collections import defaultdict


class DataHolder:

    def __init__(self):
        self.all_nodes = []
        self.features = ["curb_height_max", "obstacle_free_width_float"]
        self.point_id_map = dict()
        self.all_arcs = []

        self.row_data = dict()

        self.all_feasible_arcs = defaultdict(list)
        self.all_infeasible_arcs = defaultdict(list)
        self.all_feasible_both_way = defaultdict(list)
        self.all_infeasible_both_way = defaultdict(list)

        self.all_feasible_dir_arcs = defaultdict(list)
        self.all_infeasible_dir_arcs = defaultdict(list)

        self.M: float
        self.visual_detail_info = dict()

        """foil"""
        self.foil_route_arcs = []

        self.foil_must_feasible_arcs = []

        """fact"""
        self.fact_common_from_node_arcs = dict()
        self.fact_common_to_node_arcs = dict()

        self.foil_fact_fork_merge_nodes = dict()

        self.start_node_id = 0
        self.end_node_id = 0

        self.start_node_lc = ()
        self.end_node_lc = ()

    def get_row_info_by_arc(self, i, j):
        return self.row_data.get((i, j), self.row_data.get((j, i), None))
