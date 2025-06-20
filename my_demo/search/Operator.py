# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/19 16:22
@project:    CRC25
"""
from my_demo.search.ArcModifyTag import ArcModifyTag


class Operator:
    def __init__(self, solver):
        self.solver = solver
        self.data_holder = self.solver.data_holder

    def do_foil_must_be_feasible(self):
        for i, j in self.data_holder.foil_must_feasible_arcs:
            modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_FE)
            solution_row = self.solver.current_solution_map.loc[modified_row.name]
            modified_row['modified'] = modified_row['modified'] + solution_row['modified']
            self.solver.current_solution_map.loc[modified_row.name] = modified_row

    def do_must_be_infeasible_arcs(self):
        """公共节点 但是分叉 按说要断掉？ 但是好像只需要断一侧"""
        for from_node, arc_list in self.data_holder.fact_common_from_node_arcs.items():
            for i, j in arc_list:
                modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
                solution_row = self.solver.current_solution_map.loc[modified_row.name]
                modified_row['modified'] = modified_row['modified'] + solution_row['modified']
                self.solver.current_solution_map.loc[modified_row.name] = modified_row

        for to_node, arc_list in self.data_holder.fact_common_to_node_arcs.items():
            for i, j in arc_list:
                modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
                solution_row = self.solver.current_solution_map.loc[modified_row.name]
                modified_row['modified'] = modified_row['modified'] + solution_row['modified']
                self.solver.current_solution_map.loc[modified_row.name] = modified_row

    def make_infeasible_tail_arcs(self, problem):
        i, j = problem.sub_fact[-2], problem.sub_fact[-1]
        modified_row = self.solver.modify_df_arc_with_attr(i, j, ArcModifyTag.TO_INFE)
        solution_row = problem.map_df.loc[modified_row.name]
        modified_row['modified'] = modified_row['modified'] + solution_row['modified']
        problem.map_df.loc[modified_row.name] = modified_row

        return modified_row
