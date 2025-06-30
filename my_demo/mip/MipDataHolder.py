# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/6 14:37
@project:    CRC25
"""
from collections import defaultdict
class MipDataHolder:
    def __init__(self):
        self.all_nodes = []
        self.features = ["curb_height_max", "obstacle_free_width_float"]
        self.point_id_map = dict()
        self.all_arcs = []
        self.foil_route_arcs = []
        self.row_data = dict()

        self.all_feasible_arcs = defaultdict(list)
        self.all_infeasible_arcs = defaultdict(list)
        self.all_feasible_both_way = defaultdict(list)
        self.all_infeasible_both_way = defaultdict(list)

        self.all_feasible_dir_arcs = defaultdict(list)
        self.all_infeasible_dir_arcs = defaultdict(list)

        self.M: float
        self.visual_detail_info = dict()
    def reset_data(self):
        MipDataHolder.all_nodes = []
        MipDataHolder.features = ["curb_height_max", "obstacle_free_width_float"]
        MipDataHolder.point_id_map = dict()
        MipDataHolder.all_arcs = []
        MipDataHolder.foil_route_arcs = []
        MipDataHolder.row_data = dict()
        MipDataHolder.all_feasible_arcs = defaultdict(list)
        MipDataHolder.all_infeasible_arcs = defaultdict(list)
        MipDataHolder.all_feasible_both_way = defaultdict(list)
        MipDataHolder.all_infeasible_both_way = defaultdict(list)
        MipDataHolder.all_feasible_dir_arcs = defaultdict(list)
        MipDataHolder.all_infeasible_dir_arcs = defaultdict(list)
        MipDataHolder.M:float
        MipDataHolder.visual_detail_info=dict()