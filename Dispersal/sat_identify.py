# -*- coding: utf-8 -*-

import os  # 文件和路径操作
import json  # JSON 读写
import time  # 时间测量
import multiprocessing  # 多进程支持
from tqdm import tqdm  # 进度条显示
import pandas as pd  # 数据处理
from collections import Counter  # 计数器工具
import glob  # 文件模式匹配
from utils import get_raster_information, Release_list_element, mkdir, lcp_length, set_inf_in_list, isexist
from creat_graph import get_nodes_and_edges, creat_graph
from Find_targetCell import find_target_id
from loguru import logger
from graph_tool.all import load_graph
import warnings
warnings.filterwarnings("ignore")


def _flatten(list_of_lists):
    """辅助函数：将嵌套列表展开"""
    return [item for sublist in list_of_lists for item in sublist]


def _process_segment(args):
    """Worker函数：处理单个分块并返回该分块的 miss_count"""
    j, args_for_findID, final_table, per_cell_nums, set_segs, temp, neighbors, resolution, rows, cols = args

    # 计算当前分块的起止索引
    n = j * per_cell_nums
    m = min((j + 1) * per_cell_nums, len(final_table))
    LCPs_nodes = []
    LCPs_index = []
    miss_count = [0] * len(args_for_findID['conditions'])  # 初始化该分块的缺失计数数组

    logger.info(f"Processing segment {j+1}/{set_segs}")
    # 遍历分块单元并更新 miss_count
    for i in tqdm(range(n, m), desc=f"Segment {j+1}", ncols=80):
        result = find_target_id(num=i, **args_for_findID)
        final_table.loc[i, 'tid'] = result[0]
        final_table.loc[i, 'Sum_cost'] = result[1]
        path_list = result[3]
        # 按索引累加缺失计数
        miss_count = [x + y for x, y in zip(miss_count, result[4])]

        if not isinstance(path_list, int):
            # 计算路径长度并存储
            if neighbors == 4:
                length_val = len(path_list[0]) * resolution
            else:
                length_val = lcp_length(path_list=path_list[0], resolution=resolution, rows=rows, cols=cols)
            final_table.loc[i, 'length'] = length_val
            # 收集路径节点信息
            for l in range(len(path_list)):
                LCPs_nodes.append(path_list[l])
                LCPs_index.append(f'LCP_{j}_{i}_{l}')

    # 写出分块结果文件
    with open(os.path.join(temp, f'LCPs_list{j}.json'), 'w') as f:
        json.dump(LCPs_nodes, f)
    with open(os.path.join(temp, f'LCPs_index{j}.json'), 'w') as f:
        json.dump(LCPs_index, f)
    # 统计并写出节点出现次数
    flattened = Release_list_element(LCPs_nodes)
    counts = dict(Counter(flattened))
    with open(os.path.join(temp, f'Seg_nodes_{j}.txt'), 'w') as f:
        for k, v in counts.items():
            f.write(f"{k} {v}\n")
    # 导出分块结果表格
    final_table.iloc[n:m].to_csv(os.path.join(temp, f'Seg_final_table{j}.csv'), index=False)

    return miss_count  # 返回该分块的缺失计数数组


def sat_identify(work_dir, set_segs, start_seg=None, num_processes=None):
    """
    为每个像元匹配最优相似像元，多进程版，统计所有分块的 miss_count 并合并
    :param work_dir: 项目文件夹
    :param set_segs: 切分的片段数
    :param start_seg: 起始分块，默认0（或None）
    :param num_processes: 使用的进程数，默认全部CPU
    """
    # 1. 读取配置
    with open(os.path.join(work_dir, "configs.json"), "r") as f:
        configs = json.load(f)
    neighbors = configs["neighbors"]
    num_of_optimal_SATs = configs['num_of_optimal_SATs']

    # 2. 准备临时与日志目录
    temp = configs.get('temp') or mkdir(work_dir, folder_name='temp')
    log = configs.get('log') or mkdir(temp, folder_name='log')
    logger.add(os.path.join(log, 'S2_search_target{time}.log'), encoding="utf-8", enqueue=True)

    # 3. 构建或加载Graph
    if os.path.exists(configs['G']):
        G = load_graph(configs['G'])
        index_dict = configs['Gnode_index']
        resolution, rows, cols = configs['resolution'], configs['rows'], configs['cols']
    else:
        rows, cols, resolution, weight_list = get_raster_information(rater_path=configs['resistance_raster_path'])
        inf_list = set_inf_in_list(lst=weight_list, set_inf_criteria='<', set_inf_value=1, square=True)
        G_edges = get_nodes_and_edges(neighbors=neighbors, rows=rows, cols=cols, edge_weight_list=inf_list)
        G, index_dict = creat_graph(tool="graph_tool", rows=rows, cols=cols, G_edges=G_edges)

    # 4. 加载基础表与目标表
    base_infor = pd.read_csv(os.path.join(temp, "Base_infor.csv"))
    initial_table = pd.read_csv(os.path.join(temp, "Target_region.csv"))

    args_for_findID = {
        'G': G,
        'base_infor': base_infor,
        'final_table': initial_table,
        'index_dict': index_dict,
        'id_name': 'id',
        'conditions': configs['conditions'],
        'cons_similarity': configs['cons_similarity'],
        'compare_to_future': configs['compare_to_future'],
        'essential_for_niche': configs['essential_for_niche'],
        'top': num_of_optimal_SATs
    }

    # 5. 分块与多进程设置
    start_seg = (start_seg - 1) if start_seg else 0
    per_cell_nums = int(len(initial_table) / set_segs) + 1
    num_processes = num_processes or multiprocessing.cpu_count()
    logger.info(f"使用 {num_processes} 个进程计算，共拆分 {set_segs} 段，每段 {per_cell_nums} 个单元")

    # 6. 准备任务列表
    common_args = (args_for_findID, initial_table, per_cell_nums, set_segs, temp, neighbors, resolution, rows, cols)
    tasks = [(j, *common_args) for j in range(start_seg, set_segs)]

    # 7. 并行执行并收集各分块的 miss_count
    t1 = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        miss_lists = pool.map(_process_segment, tasks)  # 收集所有分块的缺失计数数组
    t2 = time.time()

    # 8. 合并分块结果表格并统计整体 miss_count
    # 合并 Seg_final_table CSVs
    seg_files = glob.glob(os.path.join(temp, 'Seg_final_table*.csv'))  # 查找所有分块结果
    merged_dfs = [pd.read_csv(f) for f in seg_files]
    final_table = pd.concat(merged_dfs, ignore_index=True)  # 最终合并表
    
    # 按索引位置汇总 miss_count 数组
    total_miss = [sum(x) for x in zip(*miss_lists)]  # 将每个分块的 miss_count 累加

    # 9. 日志统计与结束
    same_id_count = (final_table['id'] == final_table['tid']).sum()
    logger.info(f"所有分块处理完成，用时 {t2-t1:.2f} 秒")
    logger.info(f"每个条件缺失总数：{total_miss}")  # 输出汇总的缺失计数
    logger.info(f"原位比例：{(same_id_count/len(final_table))*100:.2f}%")
    logger.info("\n生态位变量{}因值缺失或条件约束严格, 无法匹配相似单元的百分比分别为{}".format(configs['conditions'], [f"{(num / len(final_table)) * 100:.2f}%" for num in total_miss]))

if __name__ == '__main__':
    work_dir = '../example'
    sat_identify(work_dir, set_segs=4, start_seg=None, num_processes=4)
