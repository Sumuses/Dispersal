# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:54:07 2023
@author: Su,Jie
Email: sujienju@163.com
This script is to find the target image (end point) for each image cell (start point) in the target region
that satisfies certain conditions
"""

import warnings

warnings.filterwarnings("ignore")
from graph_tool.all import *
import graph_tool as gt
import pandas as pd
import time
import numpy as np
"""
def find_target_id0(num, G, base_infor, final_table, id_name, conditions, f_conditions, cons_similarity,
                    compare_to_future, base_radius, max_radius, radius_step, direct_nodes):
    # t1 = time.time()
    one_source = final_table.loc[num]
    if one_source[conditions].isnull().any():  # 检查是否空值
        # print("ID为{}，因起始像元属性存在空值, 无法匹配相似气候单元".format(one_source[id_name]))
        return 0, 0, 1
        pass
    else:
        if len(conditions) == 0:
            print("Please identify similar conditions for climate ecological niches, with at least one")
        else:
            for i in range(len(conditions)):
                exec(f'con_{i} = int(condition_value)')  # 取出起始像元条件i的值
            for i in range(len(conditions)):
                if ("-inf" in cons_sim) | ("inf" in cons_sim) | (
                        len(cons_sim) == 1):  # 如果条件是小于, 大于, 或者 是必须相等
                    pass
                else:
                    ##########创建范围变量
                    exec(
                        f'Con{i}_nums = list(range(1+cons_sim[0], 1+cons_sim[1]+1))')  # 取出起始像元条件i的值 所对应的目标像元条件i的值可能区间范围

        ###初筛复合范围变量(条件必须满足某区间)的目标
        not_interval_con_num = []  # 记录非区间变量序号
        base_infor_c = base_infor
        for i in range(0, len(conditions)):
            con_nums_var = 'Con' + str(i) + '_nums'
            if con_nums_var in vars():  # 是否在条件变量内
                if compare_to_future[i] == 1:
                    field_name = f_condition_name
                else:
                    field_name = condition_name
                temp_base_infor_c = base_infor_c
                base_infor_c = base_infor_c.loc[
                    base_infor_c[field_name].isin(eval(con_nums_var))]  # 在基础信息表中筛选出满足范围条件的记录（潜在目标像元）

                if len(base_infor_c) > 0:
                    pass
                else:
                    base_infor_c = temp_base_infor_c  # 如果循环中间出现base_infor_C的记录为零行,则将base_infor_C返回到有值的上一个状态
            else:
                not_interval_con_num.append(i)

        if len(base_infor_c) > 0:
            if len(not_interval_con_num) > 0:
                for i in range(0, len(not_interval_con_num)):
                    if "-inf" in cons_similarity[not_interval_con_num[i]]:
                        base_infor_new = base_infor_c[base_infor_c[conditions[not_interval_con_num[i]]]
                                                      <= one_source[conditions[not_interval_con_num[i]]]]
                    elif "inf" in cons_similarity[not_interval_con_num[i]]:
                        base_infor_new = base_infor_c[base_infor_c[conditions[not_interval_con_num[i]]]
                                                      >= one_source[conditions[not_interval_con_num[i]]]]
                    elif len(cons_similarity[not_interval_con_num[i]]) == 1:

                        base_infor_new = base_infor_c[base_infor_c[conditions[not_interval_con_num[i]]].isin(
                            cons_similarity[not_interval_con_num[i]])]
                    else:
                        pass
                    if len(base_infor_new) > 0:
                        pass  # print("去掉非区间条件的不满足要求的行".format(len(base_infor)))
                    else:
                        base_infor_new = base_infor_c
            else:
                base_infor_new = base_infor  # print("无法满足非区间添加的潜在目标".format(len(base_infor)))
            source_id = int(one_source[id_name])  #####起点ID
            potential_target_id = base_infor_new[id_name].values_host  ####符合条件的ID
            # t2 = time.time()
            # print("\n{0}筛选用时{1}秒".format(source_id, t2 - t1))
            return_list = find_target_tid(G, source_id, potential_target_id, base_radius, max_radius, radius_step,
                                          direct_nodes)
            # t3 = time.time()
            # print("\n{0}匹配用时{1}秒".format(source_id, t3 - t2))
            return return_list
        else:
            return 0, 0, 5


## 在一定窗口中找到加权成本距离最小的目标像元, find_target_tid函数用时约0.8s左右
# import cugraph
def find_target_tid(G, source_id, potential_target_id, base_radius, max_radius, radius_step, direct_nodes):
    potential_target_ids = 0  # 符合条件的且 在一定窗口下的ID
    radius = base_radius

    ##以最小的窗口找到ID: potential_target_ids
    while potential_target_ids == 0 and radius <= max_radius:
        cuG2 = cugraph.ego_graph(G, source_id, radius=radius, center=True, undirected=False, distance=None)
        potential_target_id = [int(i) for i in potential_target_id]
        potential_target_ids = len(list(set(cuG2.nodes().values_host) & set(potential_target_id)))
        radius = radius + radius_step

    ##筛选出满足成本距离最小的ID: potential_target_ids_inwindow
    if potential_target_ids > 0:
        if direct_nodes:
            shortest_paths = cugraph.shortest_path(cuG2, source=source_id)  # 约慢0.3s
        else:
            shortest_paths = cugraph.shortest_path_length(cuG2, source=source_id)
        shorted_paths2 = shortest_paths.loc[shortest_paths['vertex'].isin(potential_target_id)]
        cost_path_length = min(shorted_paths2['distance'].values_host)  #
        final_path = shorted_paths2.loc[shorted_paths2['distance'] == cost_path_length]

        final_path = final_path[0:1]
        final_target_id = int(final_path['vertex'].values_host)
        if direct_nodes:
            final_path_nodes = cugraph.utils.get_traversed_path(shortest_paths, final_target_id)['vertex']  ####返回节点列表
            return final_target_id, cost_path_length, 3, final_path_nodes
        else:
            # print("{}有目标".format(source_id))
            return final_target_id, cost_path_length, 2

    else:
        return 0, 0, 4
"""

# 以下通过Graph-tool实现


def find_target_id(num, G, index_dict, base_infor, final_table, id_name, conditions,  cons_similarity,
                    compare_to_future, top, essential_for_niche):
    """
    搜索满足条件的移动成本最低的目标单元的ID，并返回其ID号，最小成本距离，标识码，最小成本路径(列表)，和缺失统计
    ：param num:在目标表格中的编号
    ：param G: 图
    ...
    """
    miss_count = [0] * len(conditions) 
    potential_target_id, source_id, identify_num = find_target_according_similarity(num, base_infor, final_table, id_name,
                                                                           conditions,  cons_similarity,
                                                                           compare_to_future,essential_for_niche,miss_count)
    if identify_num == 1 or identify_num == 3:
        return 0, 0, 0, 0,miss_count
    else:
        ID_in_baseinfor, shortest_CWD, identify_num, LCP_lists = find_target_with_CWD(G, source_id, index_dict,
                                                                                      potential_target_id, top)
        return ID_in_baseinfor, shortest_CWD, identify_num, LCP_lists, miss_count


def find_target_according_similarity(num, base_infor, final_table, id_name, conditions, cons_similarity,
                            compare_to_future, essential_for_niche, miss_count):
    """
    不考虑累积阻力成本最低, 筛选潜在的目标像元
    :param num: 在目标表格中的序号
    :param base_infor: 基础信息表
    :param final_table: 目标信息表
    :param id_name: 像元的ID字段，从1开始编号
    :param conditions: 当今的条件字段名，数组如：['c_con1', 'c_con2', 'c_con3', 'con4']
    :param cons_similarity: 上述条件所对应的相似性判别方式，数据如： [[-20, 20], [-40, 40], [-80, 80], ['-inf', 0],[0] 相同,[0,11,11]表示两个类型变量
    :param compare_to_future: 上述条件是否与未来做比较，数组如： [1, 1, 1, 0]，1表示比较，0表示不做比较
    :miss_count: 计数器
    :return:
    """
    one_source = final_table.loc[num]
    source_id = int(one_source[id_name])

    if len(conditions) == 0:    #  判断是否设置了条件
        print("Please identify similar conditions for climate ecological niches, with at least one")
        return 0, 0, 1
    else:
        base_infor_c = base_infor
        # 遍历每个条件，并从中筛选出满足条件的格网单元
        
        for index, condition_name in enumerate(conditions):   # range(len(conditions))
            match_name = compare_to_future[index] if compare_to_future[index] else condition_name

            # 取出起始像元条件i的值
            condition_value = one_source.get(condition_name, None)
            if pd.isna(condition_value) or np.isinf(condition_value):
                miss_count[index] += 1
                if essential_for_niche[index]:
                    return 0,0,3
                continue
            # 提取该变量的相似性条件
            cons_sim = cons_similarity[index]

            if cons_sim == [0]:
                base_infor_temp = base_infor_c.loc[base_infor_c[match_name]==condition_value]
                    
            elif len(cons_sim) ==2:

                if "-inf" in cons_sim:
                    base_infor_temp = base_infor_c[base_infor_c[condition_name]<= condition_value]
                elif "inf" in cons_sim:
                    base_infor_temp = base_infor_c[base_infor_c[condition_name]>= condition_value]
                else:
                    try:
                        con_list =list(range(int(cons_sim[0]+condition_value), int(cons_sim[1]+1+condition_value)))
                    except  Exception as e:
                        print(f"发生错误: {num,cons_sim[0],condition_value,cons_sim[1]}")  
                        print(f"发生错误: {e}") 
                    base_infor_temp = base_infor_c.loc[base_infor_c[match_name].isin(con_list)]
            else:
                con_list = cons_sim
                base_infor_temp = base_infor_c.loc[base_infor_c[match_name].isin(con_list)]

            # 检查 base_infor_temp 的有效性        
            if len(base_infor_temp)>0:         # ('base_infor_temp' in locals()) & 
                base_infor_c = base_infor_temp                  
            else:
                miss_count[index] = miss_count[index]+1
                if essential_for_niche[index]:              #  如果不是必须变量，则跳过该变量
                    return 0, 0, 3
                         
                
    # 提取符合条件的ID
    potential_target_id = base_infor_c[id_name] 
    # 返回输出值 
    return list(potential_target_id), source_id, 2       


def find_target_with_CWD(G, source_id, index_dict, potential_target_id, top):
    """
    在考虑累积成本最低的情况下，搜索最优目标
    :param G: graph 网络
    :param source_id: 起始像元的ID,所有表格中的ID简称ID, graph中的ID简称GID
    :param index_dict: 像元的ID在graph 网络中的对应关系,即 GID
    :param potential_target_id: fun(find_target_without_CWD)通过其他条件筛选之后的潜在目标像元ID
    :param top: 需要保留的最优路径数量
    :return: 最优目标ID, 最低成本，输出位置索引，路径list

    index_dict:
    ID  GID
    '1'  0
    '2'  1
    '4'  2

    inversion_index_dict:
    GID  ID
     0   '1'
     1  '2'
     2  '4'

    """
    # 键，值互换, 数字为Graph ID编号,"1"为原图形编号
    inversion_index_dict = {v: k for k, v in index_dict.items()}

    # 将原像元编号 转为 Graph 编号, 简称GID
    if str(source_id) in index_dict.keys():
        source = index_dict[str(source_id)]
    else:
        return 0, 0, 5, 0
    potential_target_gids = []
    for i in potential_target_id:
        if str(i) in index_dict.keys():
            one_g_id = index_dict[str(i)]
            potential_target_gids.append(one_g_id)
        else:
            pass

    if len(potential_target_gids)>0:

        # 计算起点像元至所有潜在目标像元的最小成本,并按序存在列表 all_length中
        all_length = gt.topology.shortest_distance(G, source=source, target=potential_target_gids,weights=G.ep.weight)
        # 将numpy.array 转为list
        all_length = all_length.tolist()

        # 将all_length元素从小到大排序,
        shortest_CWD = sorted(all_length)[:top]

        # 取最小的几个值在all_length中的索引,即在potential_target_gids,potential_target_id中的索引
        three_index = []
        for distance in shortest_CWD:
            three_index = three_index + get_indexes(all_length, distance)

        # 删除three_index中的重复元素
        three_index = list(set(three_index))
        
        
    else:
        return 0, 0, 5, 0

    all_LCP_lists = []

    if top == 1:
        target=potential_target_gids[three_index[0]]
        if target == source:
            return source_id, 0, 2, 0
        else:
            vlist, elist = gt.topology.shortest_path(G, source=source, target=target, weights=G.ep.weight)

            # print(f'字符{source_id}变成{source},到{target}路径的列表编号是{vlist}')
            # 提取Graph中的ID列表
            node_list = [int(v) for v in vlist]
            # 转化为原始ID列表,为短整型
            node_list = [int(inversion_index_dict[vv]) for vv in node_list]
            tid_in_baseinfor = potential_target_id[three_index[0]]
            
            all_LCP_lists.append(node_list)

            '''
            print("\n目标索引{0}".format(three_index))
            print("\n{0}--{1}".format(source_id, source))
            print("\n{0}++{1}".format(node_list[-1], tid_in_baseinfor))
            print("\n{0}".format(node_list[-1] == tid_in_baseinfor))
            print(node_list)
            '''
            # 返回最优目标原始ID, 最少累计阻力,输出位置标识,最短路径节点列表
            return tid_in_baseinfor, shortest_CWD[0], 2, all_LCP_lists

    # 当保留的最优目标不为1时，需要保留多个目标时使用
    elif top > 1:
        for index in three_index:
            vlist, elist = gt.topology.shortest_path(G, source=source, target=potential_target_gids[index],
                                                     weights=G.ep.weight)
            node_list = [int(v) for v in vlist]  ###提取Graph中的ID列表
            node_list = [int(inversion_index_dict[vv]) for vv in node_list]  ###转化为原始ID列表,为短整型
            all_LCP_lists.append(node_list)
        id_in_baseinfor = potential_target_id[three_index[0]]
        return id_in_baseinfor, shortest_CWD[0], 2, all_LCP_lists
    else:
        print("\n参数top必须为正整数,现在是{0}".format(top))


# 获取值在列表中所对应的索引
def get_indexes(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]
