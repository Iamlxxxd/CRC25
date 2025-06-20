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
import pandas as pd


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


def visual_map_foil_modded(visual_dict: dict, file_path, tag):
    import geopandas as gpd
    import branca
    import folium

    org_map_df = visual_dict.get("org_map_df")
    df_path_foil = visual_dict.get("df_path_foil")
    meta_map = visual_dict.get("meta_map")
    best_route = visual_dict.get("best_route")

    # 只画主网络
    m = org_map_df.explore(
        color="lightgrey",
        tiles="CartoDB Voyager",
        style_kwds=dict(weight=2),
        legend=False,
        name="base"
    )

    foil_arcs = list(df_path_foil['arc'])
    arc2idx = {str(arc): idx for idx, arc in enumerate(foil_arcs)}
    arc2idx.update({str((arc[1], arc[0])): idx for idx, arc in enumerate(foil_arcs)})
    org_map_df = org_map_df.copy()
    org_map_df['in_foil'] = org_map_df['arc'].apply(lambda x: str(x) in arc2idx)
    org_map_df['modded'] = org_map_df['modified'].apply(lambda x: x is not None and len(x) > 0)

    # 画整个路网 没改过的边
    not_in_foil = org_map_df[~org_map_df['modded']]
    if not not_in_foil.empty:
        m = not_in_foil.explore(
            m=m,
            color="grey",
            style_kwds=dict(weight=4),
            name="not modified",
            legend=False
        )
    # 画fact route
    df_path_fact = visual_dict.get("df_path_fact")
    df_path_fact = df_path_fact.set_crs(meta_map['CRS'])
    df_path_fact = df_path_fact.to_crs(org_map_df.crs)
    if not df_path_fact.empty:
        m = df_path_fact.explore(
            m=m,
            color="black",
            style_kwds=dict(weight=7),
            name="fact route",
            legend=False,
            layer_kwds={"show": True, "overlay": True, "control": True, "group": "fact route"}
        )

    if not best_route.empty:
        best_route = best_route.set_crs(meta_map['CRS'])
        best_route = best_route.to_crs(org_map_df.crs)
        m = best_route.explore(m=m,
                               color="#00A1FF",
                               style_kwds=dict(weight=10),
                               name="best route",
                               legend=False,
                               layer_kwds={"show": True, "overlay": True, "control": True, "group": "best route"}
                               )

    # 画所有modded且不在foil_path的边（橘红色，宽度7）
    modded_not_in_foil = org_map_df[(org_map_df['modded']) & (~org_map_df['in_foil'])]
    if not modded_not_in_foil.empty:
        m = modded_not_in_foil.explore(
            m=m,
            color="red",
            style_kwds=dict(weight=7),
            name="modded not in foil",
            legend=False,
            layer_kwds={"show": True, "overlay": True, "control": True, "group": "modded not in foil"}
        )

    # ----------- 图层4：Foil Route（交替黑/橙/红） -----------
    foil_edges = org_map_df[org_map_df['in_foil']].sort_values(by='arc', key=lambda x: x.map(arc2idx))
    foil_edges = foil_edges.copy()
    color_list = []
    last_color = None
    for _, row in foil_edges.iterrows():
        if row['modded']:
            color = 'orange' if last_color != 'orange' else 'red'
            last_color = color
        else:
            color = '#F5A623'
        color_list.append(color)
    foil_edges['color'] = color_list

    foil_group = folium.FeatureGroup(name="Foil Route", show=True)
    # 获取所有字段名，常用如 arc, modified
    tooltip_fields = [col for col in foil_edges.columns if col not in ['geometry']]
    for _, row in foil_edges.iterrows():
        geo = gpd.GeoDataFrame([row], crs=org_map_df.crs)
        folium.GeoJson(
            geo,
            style_function=lambda x, color=row['color']: {"color": color, "weight": 6},
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)
        ).add_to(foil_group)
    foil_group.add_to(m)

    # 添加LayerControl控件
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="position: fixed; 
                top: 45px; right: 10px; width: 200px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: 0.85;">
      <b>color of line</b><br>
      <span style="display:inline-block;width:20px;height:4px;background:grey;margin-right:5px;"></span>not modified<br>
      <span style="display:inline-block;width:20px;height:4px;background:black;margin-right:5px;"></span>fact route<br>
      <span style="display:inline-block;width:20px;height:4px;background:#00A1FF;margin-right:5px;"></span>best route<br>
      <span style="display:inline-block;width:20px;height:4px;background:red;margin-right:5px;"></span>modded not in foil<br>
      <span style="display:inline-block;width:20px;height:4px;background:orange;margin-right:5px;"></span>Foil Route<br>
    </div>
    """

    # 新增：根据show_data生成图例
    show_data = visual_dict.get("show_data")
    if show_data:
        show_data_html = '<div style="position: fixed; top: 260px; right: 10px; width: 220px; height: auto; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: 0.85; padding: 8px;">'
        show_data_html += "<b>Data Info</b><br>"
        for k, v in show_data.items():
            show_data_html += f"<span style='font-weight:bold'>{k}:</span> {v}<br>"
        show_data_html += "</div>"
        legend_html += show_data_html

    m.get_root().html.add_child(branca.element.Element(legend_html))
    m.save(os.path.join(file_path, f"{tag}_visual.html"))
    return m


def visual_sub_problem(visual_dict: dict, file_path, tag):
    import geopandas as gpd
    import branca
    import folium

    org_map_df = visual_dict.get("org_map_df")
    meta_map = visual_dict.get("meta_map")
    sub_fact = visual_dict.get("sub_fact")
    sub_foil = visual_dict.get("sub_foil")
    sub_best = visual_dict.get("sub_best")
    # 只画主网络
    m = org_map_df.explore(
        color="lightgrey",
        tiles="CartoDB Voyager",
        style_kwds=dict(weight=7),
        legend=False,
        name="base"
    )

    sub_best = sub_best.set_crs(meta_map['CRS'])
    sub_best = sub_best.to_crs(org_map_df.crs)
    if not sub_best.empty:
        sub_best = sub_best.set_crs(meta_map['CRS'])
        sub_best = sub_best.to_crs(org_map_df.crs)
        m = sub_best.explore(m=m,
                             color="#00A1FF",
                             style_kwds=dict(weight=10),
                             name="best route",
                             legend=False,
                             layer_kwds={"show": True, "overlay": True, "control": True, "group": "best route"}
                             )

    if not sub_fact.empty:
        m = sub_fact.explore(
            m=m,
            color="black",
            style_kwds=dict(weight=7),
            name="sub fact",
            legend=False,
            layer_kwds={"show": True, "overlay": True, "control": True, "group": "sub fact"}
        )
    if not sub_foil.empty:
        m = sub_foil.explore(
            m=m,
            color="#F5A623",
            style_kwds=dict(weight=6),
            name="sub foil",
            legend=False,
            layer_kwds={"show": True, "overlay": True, "control": True, "group": "sub foil"}
        )

    # 新增：根据show_data生成图例
    legend_html = """
    <div style="position: fixed; 
                top: 45px; right: 10px; width: 200px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: 0.85;">
      <b>color of line</b><br>
      <span style="display:inline-block;width:20px;height:4px;background:lightgrey;margin-right:5px;"></span>base<br>
      <span style="display:inline-block;width:20px;height:4px;background:black;margin-right:5px;"></span>sub fact<br>
      <span style="display:inline-block;width:20px;height:4px;background:#00A1FF;margin-right:5px;"></span>best route<br>
      <span style="display:inline-block;width:20px;height:4px;background:#F5A623;margin-right:5px;"></span>sub foil<br>
    </div>
    """
    show_data = visual_dict.get("show_data")
    if show_data:
        show_data_html = '<div style="position: fixed; top: 260px; right: 10px; width: 220px; height: auto; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity: 0.85; padding: 8px;">'
        show_data_html += "<b>Data Info</b><br>"
        for k, v in show_data.items():
            show_data_html += f"<span style='font-weight:bold'>{k}:</span> {v}<br>"
        show_data_html += "</div>"
        legend_html += show_data_html

    m.get_root().html.add_child(branca.element.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(file_path, f"{tag}_visual.html"))
