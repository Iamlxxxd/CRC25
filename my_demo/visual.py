# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/29 10:46
@project:    CRC25
"""

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def visual_line(solver):

    each_iter_best_individual = solver.each_iter_best_individual

    objs = [ind.obj for ind in each_iter_best_individual]
    graph_errors = [ind.graph_error for ind in each_iter_best_individual]
    route_errors = [ind.route_error for ind in each_iter_best_individual]

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
    #todo save
    plt.show()


def visual_map(solver):
    gdf_coords = solver.config.gdf_coords_loaded
    origin_node_loc_length = solver.origin_node_loc
    dest_node_loc_length = solver.dest_node_loc
    meta_map = solver.meta_map
    df_path_fact = solver.df_path_fact
    df_path_foil = solver.df_path_foil
    best_route = solver.best_individual.path_df

    # Subset network for plotting
    my_rad = 70
    gdf_coords['buffer'] = gdf_coords['geometry'].buffer(my_rad, cap_style=3)
    plot_area = gpd.GeoDataFrame(geometry=[gdf_coords['buffer'][0].union(gdf_coords['buffer'][1])], crs=meta_map["CRS"])
    df_sub = gpd.sjoin(solver.org_map_df, plot_area, how='inner').reset_index()

    attrs_color = {"path_type": {"c": "yellow", "ls": "-", "lw": 5},
                   "curb_height_max": {"c": "green", "ls": "-", "lw": 4},
                   "obstacle_free_width_float": {"c": "orange", "ls": "-", "lw": 3}}
    fig, ax = plt.subplots(figsize=(12, 12))

    # Network
    df_sub.plot(ax=ax, color='lightgrey', linewidth=1)

    df_path_fact.plot(ax=ax, color='grey', linewidth=4)
    df_path_foil.plot(ax=ax, color='black', linewidth=4)
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
    #todo save
    plt.show()