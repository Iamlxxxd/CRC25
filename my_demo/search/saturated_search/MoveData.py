# encoding:utf-8
"""
@email  :    xiangdong.lv@geekplus.com
@auther :    XiangDongLv
@time   :    2025/6/28 09:47
@project:    CRC25
"""


class MoveData:
    def __init__(self, arc, modify_tag, operator_type):
        self.arc = arc
        self.modify_tag = modify_tag
        self.operator_type = {operator_type}

    @property
    def sort_key(self):
        """返回用于排序的元组"""
        return (self.arc, self.modify_tag)

    def __hash__(self):
        return hash((self.arc, self.modify_tag))

    def __eq__(self, other):
        if not isinstance(other, MoveData):
            return False
        return self.sort_key == other.sort_key

    def __lt__(self, other):
        if not isinstance(other, MoveData):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __str__(self):
        # 千万别改 problemNode用他做hash
        return f"{self.arc},{self.modify_tag.name}"

    __repr__ = __str__
