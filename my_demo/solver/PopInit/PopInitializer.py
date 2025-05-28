# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/5/23 16:50
@project:    CRC25
"""

import math

from my_demo.solver.PopInit.InitFromRandom import InitFromRandom


class PopInitializer:
    init_strategy_list = []

    def __init__(self, solver):
        self.solver = solver
        self.__set_strategy_list()

    def __set_strategy_list(self):
        self.init_strategy_list.append(InitFromRandom(self.solver))
        # todo other

    def heuristic_init_pop(self):
        pop_size = self.solver.pop_size
        gradient = int(math.ceil(pop_size / len(self.init_strategy_list)))
        left = 0
        right = gradient

        for each_strategy in self.init_strategy_list:
            if each_strategy == self.init_strategy_list[-1]:
                right = pop_size
            each_strategy.do_init_pop(left, right)
            left = right
            right = right + gradient
