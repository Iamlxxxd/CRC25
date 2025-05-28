# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/21 18:43
@project:    CRC25
"""

from copy import deepcopy

from geopandas import GeoDataFrame
from networkx.classes import MultiDiGraph

from utils.dataparser import create_network_graph, handle_weight_with_recovery


class Individual:
    # 地图数据
    org_df: GeoDataFrame

    weight_df: GeoDataFrame

    network: MultiDiGraph

    user_model: dict

    graph_error: int

    route_error: float

    obj: float

    def __init__(self, org_geo_df: GeoDataFrame, user_model: dict):
        self.org_df = org_geo_df
        self.user_model = user_model
        self.create_network_graph()

    def create_network_graph(self):
        """
        Create a network graph from the GeoDataFrame.
        """
        weight_df = deepcopy(self.org_df)
        self.weight_df = handle_weight_with_recovery(weight_df, self.user_model)
        _, self.network = create_network_graph(self.weight_df)

    def __str__(self):
        return f"obj:{self.obj},gE:{self.graph_error},rE:{self.route_error}"

    def __repr__(self):
        return self.__str__()
