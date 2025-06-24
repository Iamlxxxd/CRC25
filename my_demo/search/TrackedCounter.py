# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/24 14:10
@project:    CRC25
"""
import itertools

class TrackedCounter:
    def __init__(self, start=0, step=1):
        self._counter = itertools.count(start, step)
        self.current = start - step
        self._step = step
    def __iter__(self):
        return self

    def __next__(self):
        self.current = next(self._counter)
        return self.current

    def peek(self):
        return self.current + self._step