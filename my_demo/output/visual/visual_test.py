# encoding:utf-8
"""
@email  :    lvxiangdong2qq@163.com
@auther :    XiangDongLv
@time   :    2025/6/11 17:21
@project:    CRC25
"""
import pickle
from my_demo.visual import visual_map_explore

with open("/Users/lvxiangdong/Desktop/work/some_project/CRC25/my_demo/output/visual/temp.pkl", 'rb') as f:
    info =  pickle.load(f)

visual_map_explore(info)