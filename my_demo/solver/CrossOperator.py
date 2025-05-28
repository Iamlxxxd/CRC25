# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/22 17:00
@project:    CRC25
"""

import numpy as np
class CrossOperator:

    def __init__(self, solver):
        self.solver = solver


    def do_move(self):
        org_pop = self.solver.pop
        mut_pop = self.solver.mut_pop
        mask = np.random.rand(len(org_pop)) < self.solver.PROB_CROSS
        self.solver.cross_pop = np.where(mask, org_pop, mut_pop)