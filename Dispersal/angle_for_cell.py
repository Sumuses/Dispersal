# -*- coding: utf-8 -*-

"""
将风速风向转换成U风V风
"""
import pandas as pd
import json
from tqdm import tqdm
from utils import azimuthangle

work_dir = '../example/'
temp = work_dir + 'temp/'
output = temp + 'output/'
base_infor = pd.read_csv(work_dir + "base_infor.csv", dtype=int)
angle_for_eachcell = base_infor[['id', 'x', 'y']]
id_name = "id"


def wsd2uv(ws, wd):
    """
    根据速风向转换成U风V风
    :param ws: 风速
    :param wd: 风向
    :return:
    """
    import numpy as np
    wd = 270 - wd
    wd = wd /180 *np.pi
    x = ws * np.cos(wd)
    y = ws * np.sin(wd)
    return(x, y)

with open(temp + "LCPs_list_all.json", "r") as f:  # 打开文件
    LCPs_list = json.load(f)  # 读取文件

for lst in tqdm(LCPs_list): #不可用进程池，可能会干扰
    one_line_points = []
    for i in range(0,len(lst)-1):
        x1 = angle_for_eachcell[angle_for_eachcell[id_name] == lst[i]].x.tolist()[0]
        y1 = angle_for_eachcell[angle_for_eachcell[id_name] == lst[i]].y.tolist()[0]
        x2 = angle_for_eachcell[angle_for_eachcell[id_name] == lst[i+1]].x.tolist()[0]
        y2 = angle_for_eachcell[angle_for_eachcell[id_name] == lst[i+1]].y.tolist()[0]
        angle = azimuthangle(x1, y1, x2, y2)
        angle = angle_for_eachcell[angle_for_eachcell[id_name] == lst[i]].angle.tolist()[0]+angle
        angle_for_eachcell.loc[1,'angle'] = angle

all_final_table = pd.read_csv(output + "all_final_table.csv", dtype=int)
all_final_table = all_final_table[['id', 'num']]
