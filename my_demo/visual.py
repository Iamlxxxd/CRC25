# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/29 10:46
@project:    CRC25
"""
import os.path

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def visual_line(visual_dict: dict):
    objs = visual_dict.get("objs")
    graph_errors = visual_dict.get("graph_errors")
    route_errors = visual_dict.get("route_errors")

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(objs, marker='o', label='obj')
    axs[0].set_ylabel('obj')
    axs[0].set_title('obj')
    axs[0].set_xlabel('iter_times')

    axs[1].plot(graph_errors, marker='o', color='orange', label='graph_error')
    axs[1].set_ylabel('graph_error')
    axs[1].set_title('graph_error')
    axs[1].set_xlabel('iter_times')

    axs[2].plot(route_errors, marker='o', color='green', label='route_error')
    axs[2].set_ylabel('route_error')
    axs[2].set_title('route_error')
    axs[2].set_xlabel('iter_times')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    # todo save
    plt.show()


def visual_map(visual_dict: dict):
    gdf_coords = visual_dict.get("gdf_coords")
    origin_node_loc_length = visual_dict.get("origin_node_loc_length")
    dest_node_loc_length = visual_dict.get("dest_node_loc_length")
    meta_map = visual_dict.get("meta_map")
    df_path_fact = visual_dict.get("df_path_fact")
    df_path_foil = visual_dict.get("df_path_foil")
    best_route = visual_dict.get("best_route")

    # Subset network for plotting
    my_rad = 70
    gdf_coords['buffer'] = gdf_coords['geometry'].buffer(my_rad, cap_style=3)
    plot_area = gpd.GeoDataFrame(geometry=[gdf_coords['buffer'][0].union(gdf_coords['buffer'][1])], crs=meta_map["CRS"])
    df_sub = gpd.sjoin(visual_dict.get("org_map_df"), plot_area, how='inner').reset_index()

    attrs_color = {"path_type": {"c": "yellow", "ls": "-", "lw": 5},
                   "curb_height_max": {"c": "green", "ls": "-", "lw": 4},
                   "obstacle_free_width_float": {"c": "orange", "ls": "-", "lw": 3}}
    fig, ax = plt.subplots(figsize=(12, 12))

    # Network
    df_sub.plot(ax=ax, color='lightgrey', linewidth=1)

    df_path_fact.plot(ax=ax, color='grey', linewidth=7)
    df_path_foil.plot(ax=ax, color='black', linewidth=7)
    best_route.plot(ax=ax, color='green', linewidth=2)

    # not_common_edges_df.plot(ax=ax, color='yellow', linewidth=2)
    # Origin and destination location
    gdf_coords.head(1).plot(ax=ax, color='blue', markersize=50)
    gdf_coords.tail(1).plot(ax=ax, color='red', markersize=50)

    # Origin and destination nodes
    gpd.GeoSeries([origin_node_loc_length], crs=meta_map["CRS"]).plot(ax=ax, color='blue', markersize=20)
    gpd.GeoSeries([dest_node_loc_length], crs=meta_map["CRS"]).plot(ax=ax, color='red', markersize=20)

    # Background
    cx.add_basemap(ax=ax, source=cx.providers.CartoDB.Voyager, crs=meta_map["CRS"])

    # Legend
    route_acc = mpatches.Patch(color='black', label='foil_route')
    route = mpatches.Patch(color='grey', label='fact_route')
    route_best = mpatches.Patch(color='green', label='best_route (perturbed)')
    origin = mpatches.Patch(color='blue', label='Orgin')
    dest = mpatches.Patch(color='red', label='destination')
    legend_handles = [route_acc, route, route_best, origin, dest]
    for attr, color in attrs_color.items():
        legend_handles.append(mpatches.Patch(color=color["c"], label=attr))

    plt.legend(handles=legend_handles, loc="lower right")

    plt.axis('off')
    # todo save
    plt.show()


def visual_map_explore(visual_dict: dict,file_path):
    import geopandas as gpd
    meta_map = visual_dict.get("meta_map")
    gdf_coords = visual_dict.get("gdf_coords")
    origin_node = visual_dict.get("origin_node_loc_length")
    dest_node = visual_dict.get("dest_node_loc_length")
    df_path_fact = visual_dict.get("df_path_fact")
    df_path_foil = visual_dict.get("df_path_foil")
    best_route = visual_dict.get("best_route")
    org_map_df = visual_dict.get("org_map_df")
    config = visual_dict.get("config")

    # 只画主网络
    m = org_map_df.explore(
        color="lightgrey",
        tiles="CartoDB Voyager",
        style_kwds=dict(weight=1),
        legend=False
    )

    if df_path_fact is not None and not df_path_fact.empty and df_path_fact.geometry.notnull().any():
        df_path_fact = df_path_fact.set_crs(meta_map['CRS'])
        df_path_fact = df_path_fact.to_crs(org_map_df.crs)
        m = df_path_fact.explore(m=m, color="grey", style_kwds=dict(weight=7), name="fact_route", legend=False)


    if df_path_foil is not None and not df_path_foil.empty and df_path_foil.geometry.notnull().any():
        m = df_path_foil.explore(m=m, color="black", style_kwds=dict(weight=7), name="foil_route", legend=False)
    if best_route is not None and not best_route.empty and best_route.geometry.notnull().any():
        m = best_route.explore(m=m, color="green", style_kwds=dict(weight=2), name="best_route", legend=False)

    # 起点终点
    if gdf_coords is not None and not gdf_coords.empty:
        m = gdf_coords.head(1).explore(m=m, color="blue", marker_kwds=dict(radius=8), name="Origin", legend=False)
        m = gdf_coords.tail(1).explore(m=m, color="red", marker_kwds=dict(radius=8), name="Destination", legend=False)

    # 起点终点节点
    if origin_node is not None:
        m = gpd.GeoSeries([origin_node], crs=org_map_df.crs).explore(m=m, color="blue", marker_kwds=dict(radius=5),
                                                                 name="Origin Node", legend=False)
    if dest_node is not None:
        m = gpd.GeoSeries([dest_node], crs=org_map_df.crs).explore(m=m, color="red", marker_kwds=dict(radius=5),
                                                               name="Dest Node", legend=False)

    m.save(os.path.join(file_path,"map_f.html"))
    return m
