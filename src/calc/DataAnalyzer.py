# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 14:38
@project:    CRC25
"""

from collections import namedtuple


class DataAnalyzer:
    def __init__(self, solver):
        self.solver = solver
        self.config = solver.config
        self.data_holder = solver.data_holder
        self.df_path_foil = solver.df_path_foil
        self.df_path_fact = solver.df_path_fact

    def do_basic_analyze(self):
        self.foil_arc_must_be_feasible()
        self.find_sub_forks_and_merges_node(self.df_path_foil, self.df_path_fact, self.data_holder)

    def foil_arc_must_be_feasible(self):
        """
        foil 中的路径 如果不可行 必须转成可行

        Returns:
        """
        for idx, foil_row in self.df_path_foil.iterrows():
            i, j = foil_row['arc']
            row = self.data_holder.get_row_info_by_arc(i, j)

            if not row['curb_height_max_include'] or not row['obstacle_free_width_float_include']:
                self.data_holder.foil_must_feasible_arcs.append((i, j))

    def find_sub_forks_and_merges_node(self, df_path_foil, df_path_fact, data_holder):
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

        foil_nodes = extract_nodes(df_path_foil)
        fact_nodes = extract_nodes(df_path_fact)

        # 找到所有分叉和汇聚点
        i, j = 0, 0
        n_foil, n_fact = len(foil_nodes), len(fact_nodes)
        result = {}

        while i < n_foil and j < n_fact:
            if foil_nodes[i] == fact_nodes[j]:
                i += 1
                j += 1
            else:
                # 记录分叉点
                fork = foil_nodes[i - 1] if i > 0 else None
                foil_start, fact_start = i, j

                # 用集合找最近的交汇点
                foil_set = set(foil_nodes[foil_start:])
                fact_set = set(fact_nodes[fact_start:])
                common_nodes = foil_set & fact_set

                # 如果没有交汇点，直接跳到下一个分叉
                if not common_nodes:
                    i += 1
                    j += 1
                    continue

                # 找到最近的交汇点
                min_foil_idx = n_foil
                min_fact_idx = n_fact
                merge = None
                for node in common_nodes:
                    idx_foil = foil_nodes.index(node, foil_start)
                    idx_fact = fact_nodes.index(node, fact_start)
                    if idx_foil + idx_fact < min_foil_idx + min_fact_idx:
                        min_foil_idx = idx_foil
                        min_fact_idx = idx_fact
                        merge = node

                foil_sub_path = foil_nodes[foil_start-1:min_foil_idx+1]
                fact_sub_path = fact_nodes[fact_start-1:min_fact_idx+1]

                # 记录分叉和交汇的信息
                result[fork] = {
                    'fork': fork,
                    'merge': merge,
                    'foil_sub_path': foil_sub_path,
                    'fact_sub_path': fact_sub_path
                }

                # 更新指针，继续查找下一个分叉点
                i = min_foil_idx
                j = min_fact_idx

        data_holder.foil_fact_fork_merge_nodes = result

