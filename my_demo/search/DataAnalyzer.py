# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 14:38
@project:    CRC25
"""

from collections import namedtuple, defaultdict


class DataAnalyzer:
    def __init__(self, solver):
        self.solver = solver
        self.config = solver.config
        self.data_holder = solver.data_holder
        self.df_path_foil = solver.df_path_foil
        self.df_path_fact = solver.df_path_fact

    def do_analyze(self):
        self.arc_must_be_feasible()
        self.find_common_node_rows()
        self.find_sub_forks_and_merges_node()

    def arc_must_be_feasible(self):
        """
        foil 中的路径 如果不可行 必须转成可行

        Returns:
        """
        for idx, foil_row in self.df_path_foil.iterrows():
            i, j = foil_row['arc']
            row = self.data_holder.get_row_info_by_arc(i, j)

            if not row['curb_height_max_include'] or not row['obstacle_free_width_float_include']:
                self.data_holder.foil_must_feasible_arcs.append((i, j))

    def find_common_node_rows(self):
        """
        找出两条路径中，从同一个点出发和到同一个点到达的所有数据。
        返回:
            from_node_dict: {node: [CommonRows(foil_row, fact_row), ...]}
            to_node_dict: {node: [CommonRows(foil_row, fact_row), ...]}
        """
        from_node_dict = defaultdict(list)
        to_node_dict = defaultdict(list)

        # 构建fact的起点和终点索引
        fact_from = defaultdict(list)
        fact_to = defaultdict(list)
        for idx, row in self.df_path_fact.iterrows():
            i, j = row['arc']
            # 理论上这里应该在同一个路径上不会自己有分叉 但是防止真的有这种数据先构建成list
            fact_from[i].append(j)
            fact_to[j].append(i)

        # 从foil出发点匹配fact出发点
        for idx, foil_row in self.df_path_foil.iterrows():
            i, j = foil_row['arc']
            # 从同一个点出发
            if i in fact_from:
                for to_node in fact_from[i]:
                    if j != to_node:
                        # 从同一个点出发，但指向不同位置
                        from_node_dict[i].append((i, to_node))

            # 到同一个点到达
            if j in fact_to:
                for from_node in fact_to[j]:
                    if i != from_node:
                        # 到达同一个点，但来源不同
                        to_node_dict[j].append((from_node, j))

        self.data_holder.fact_common_from_node_arcs = from_node_dict
        self.data_holder.fact_common_to_node_arcs = to_node_dict

    def find_sub_forks_and_merges_node(self):
        """
        找出foil和fact路径在某个位置分叉又汇聚的起点和终点。
        例如: foil: a-b-c-d-e, fact: a-b-f-d-e, 记录b和d点，并记录分叉和汇聚之间的各自子路径。
        """
        ForkMerge = namedtuple('ForkMerge', ['fork', 'merge', 'foil_sub_path', 'fact_sub_path'])
        result = dict()

        # 提取foil和fact的节点序列
        def extract_nodes(df):
            nodes = []
            for idx, row in df.iterrows():
                i, j = row['arc']
                if not nodes:
                    nodes.append(i)
                nodes.append(j)
            return nodes

        foil_nodes = extract_nodes(self.df_path_foil)
        fact_nodes = extract_nodes(self.df_path_fact)

        # 找到所有分叉和汇聚点
        i, j = 0, 0
        n_foil, n_fact = len(foil_nodes), len(fact_nodes)
        while i < n_foil and j < n_fact:
            if foil_nodes[i] == fact_nodes[j]:
                i += 1
                j += 1
            else:
                fork = foil_nodes[i - 1] if i > 0 else None
                ii, jj = i, j
                while ii < n_foil and jj < n_fact and foil_nodes[ii] != fact_nodes[jj]:
                    ii += 1
                    jj += 1
                if ii < n_foil and jj < n_fact:
                    merge = foil_nodes[ii]
                    # 记录分叉和汇聚之间的子路径（不含fork和merge点）
                    foil_sub_path = foil_nodes[i:ii]
                    fact_sub_path = fact_nodes[j:jj]
                    result[fork] = ForkMerge(
                        fork=fork,
                        merge=merge,
                        foil_sub_path=foil_sub_path,
                        fact_sub_path=fact_sub_path
                    )
                    i = ii
                    j = jj
                else:
                    break

        self.data_holder.foil_fact_fork_merge_nodes = result
