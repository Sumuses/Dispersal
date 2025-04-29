# -*- coding: utf-8 -*-
"""
根据第一步输出的结果，汇总表格并统计计算基于距离的气候变化速度、速度的方向、
迁徙过程中的累积人类活动暴露量以及气候走廊得分
"""

import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from utils import azimuthangle, mkdir, get_raster_information
from RW_raster import write_to_tiff
import json


def sat_post_process(work_dir, time=None):
    temp = work_dir + '/temp'
    with open(work_dir + "/configs.json", "r") as f:  # 打开文件
        configs = json.load(f)  # 读取文件
    id_name = 'id'
    base_infor = pd.read_csv(temp + "/base_infor.csv")
    print("\n 第1/4步，汇总信息表")
    # 获取文件夹内所有目标表格的文件名
    Seg_final_tablefile = []
    
    output = mkdir(work_dir, folder_name='output')
    filename_list = os.listdir(temp)

    for filename in filename_list:
        if filename[:10] == 'Seg_final_':  #将读取文件名字的前十个字符与'Seg_final_'匹对
            Seg_final_tablefile.append(filename)
        else:
            pass

    # 检查是否为空
    if len(Seg_final_tablefile) > 0:
        print("数据保存在{0}个结果中，注意检查是否与第一步一致".format(len(Seg_final_tablefile)))
    else:
        print("未找到上一步的结果，请确保！未更改！上一步的结果的保存路径")

    # 合并表格
    all_final_table = pd.DataFrame()
    for filename in Seg_final_tablefile:
        oneSeg_final_table = pd.read_csv(temp +'/' + filename, header=0, index_col=0)
        all_final_table = pd.concat([all_final_table, oneSeg_final_table])  # .update(final_tar) 将所有数据更新上

    # 处理重复项:如果ID重复，只保留一个
    all_final_table.drop_duplicates(subset=[id_name], keep='first', inplace=True)

    # 去掉tid等于0的行
    all_final_table = all_final_table[~(all_final_table['tid'] == 0)]
    all_final_table.loc[all_final_table['id'] == all_final_table['tid'], 'length'] = 0
    
    all_final_table = all_final_table[['x', 'y', 'id', 'tid', 'Sum_cost', 'length']+configs['analyse_objects']]

    print("\n 第2/4步，计算速度、角度")
    # 根据id 匹配 tx, ty, 并计算角度
    tx = []
    ty = []
    angle = []
    for index, row in all_final_table.iterrows():
        itx = base_infor[base_infor[id_name] == row['tid']].x.tolist()[0]
        ity = base_infor[base_infor[id_name] == row['tid']].y.tolist()[0]
        iangle = azimuthangle(row['x'], row['y'], itx, ity)
        tx.append(itx)
        ty.append(ity)
        angle.append(iangle)

    all_final_table['tx'] = tx
    all_final_table['ty'] = ty
    all_final_table['angle'] = angle
    all_final_table['velocity'] = all_final_table['length']
    all_final_table.to_csv(output+'/all_final_table.csv')

    print("\n 第3/4步，统计走廊得分")
    # 统计路径重叠次数，即 走廊得分
    Seg_nodes_file = []
    for filename in filename_list:
        if filename[:10] == 'Seg_nodes_':  #将读取文件名字的前十个字符与'Seg_nodes_'匹对
            Seg_nodes_file.append(filename)
        else:
            pass

    path_nodes = pd.DataFrame()
    for filename in Seg_nodes_file:
        oneSeg_nodes = pd.read_csv(temp +'/' + filename, header=None, sep=" ")
        path_nodes = pd.concat([path_nodes, oneSeg_nodes], ignore_index=True)
    path_nodes.columns = ['id', 'num']
    path_nodes = path_nodes.groupby(['id'])['num'].sum()
    path_nodes = path_nodes.reset_index(drop=False)

    print("\n 第4/4步，生成新表格，存储为速度、累积阻力、走廊得分")
    # 为绘制栅格图做准备
    big_table = pd.merge(base_infor[[id_name]], path_nodes, how='left', on=id_name)
    big_table = pd.merge(big_table, all_final_table[[id_name, 'velocity', 'angle', 'Sum_cost']], how='left', on=id_name)
    big_table.to_csv(temp+'/table_for_map.csv')



def sat_landscape_indicator(work_dir,indicators=None, time=None):
    '''
    '''
    # 处理参数
    if indicators:
        pass
    else:
        indicators= ["dispersal velocity","cumulative exposure","corridors score"]

    with open(work_dir + "/configs.json", "r") as f:  # 打开文件
        configs = json.load(f)  # 读取文件
    output = mkdir(work_dir, folder_name='output')
    path_raster = configs.get('resistance_raster_path')
    rows = int(configs.get('rows'))
    cols = int(configs.get('cols'))
    final_table =pd.read_csv(configs['temp']+'/table_for_map.csv') 
    NoData = np.nan
    if time:
        final_table['velocity'] = final_table['velocity']/1000/time
    else:
        pass
    band1 = final_table['velocity'].fillna(NoData).values.reshape(rows, cols)
    band2 = final_table['angle'].fillna(NoData).values.reshape(rows, cols)
    band3 = final_table['Sum_cost'].fillna(NoData).values.reshape(rows, cols)
    band4 = final_table['num'].fillna(NoData).values.reshape(rows, cols)
    bands = [band1, band2]

    if "dispersal velocity" in indicators:
        write_to_tiff(src_filename=path_raster,
                    dst_filename=output+'/Dispersal_velocity.tif',    # velocity_angle_Sumcost_corridorsscore
                    bands_num=2,
                    bands_arr=bands)
    if "cumulative exposure" in indicators:
        write_to_tiff(src_filename=path_raster,
                    dst_filename=output+'/Cumulative_exposure.tif',    # velocity_angle_Sumcost_corridorsscore
                    bands_num=1,
                    bands_arr=[band3])
    if "corridors score" in indicators:
        write_to_tiff(src_filename=path_raster,
                    dst_filename=output+'/Corridors_score.tif',    # velocity_angle_Sumcost_corridorsscore
                    bands_num=1,
                    bands_arr=[band4])
    else:
        print("请dispersal velocity, corridors score, accumulative exposure 中从选择需要计算的景观指标")

    print('景观指标计算完成, 结果保存在{} 文件夹'.format(output))

def sat_patch_indicator(work_dir,patch_name, indicators=None):
    '''

    '''
    # 参数处理
    if indicators:
        pass
    else:
        indicators= ["displacement index", "disappear niches index", "novel niches index"]

    temp = work_dir + '/temp'
    with open(work_dir + "/configs.json", "r") as f:  # 打开文件
        configs = json.load(f)  # 读取文件
    
    output = mkdir(work_dir, folder_name='output')
    base_infor = pd.read_csv(temp + "/base_infor.csv")
    all_final_table = pd.read_csv(output + "/all_final_table.csv")
    # filtered_df = base_infor[base_infor['id'].isin(all_final_table['id'])]
    
    similarity = configs['cons_similarity']
    patch_names = all_final_table[patch_name].unique().tolist()
    ######## 1 根据json文件筛选出必要条件的名称 和 对应条件
    # 使用列表推导式进行筛选
    conditions = [x for x, f in zip(configs['conditions'], configs['essential_for_niche']) if f]
    f_conditions = [y if y is not None else x for x, y in zip(configs['conditions'], configs['compare_to_future'])]
    f_conditions = [x for x, f in zip(f_conditions, configs['essential_for_niche']) if f]
    similarity = [x for x, f in zip(configs['cons_similarity'], configs['essential_for_niche']) if f]

    patch_ind = pd.DataFrame()
    
    for patch in patch_names:
        one_patch ={}
        one_patch['patch_name'] = patch

        patch_df = base_infor[base_infor[patch_name] == patch]
        if "disappear niches index" in indicators:
            disa_nichle = check_nichles_in_patch (patch_df,conditions1 = conditions, conditions2 = f_conditions, similarity=similarity)
            one_patch['disappear_index'] = disa_nichle[1]

        if "novel niches index" in indicators:
            novel_nichle = check_nichles_in_patch (patch_df,conditions1 = f_conditions, conditions2 = conditions, similarity=similarity)
            one_patch['novel_index'] = novel_nichle[1]

        if "displacement index" in indicators:
            patch_df = all_final_table[all_final_table[patch_name] == patch]
            disp_count = 0
            for index, row in patch_df.iterrows():
                tid = row.get('tid', None)
                if tid in patch_df['id'].values:
                    pass
                else:
                    disp_count += 1
            disp_per = disp_count/len(patch_df)
            one_patch['displacement_index'] = disp_per

        patch_ind = pd.concat([patch_ind, pd.DataFrame([one_patch])], ignore_index=True)
        # patch_ind._append(one_patch, ignore_index =True)
    patch_ind.to_csv(output+'/patch_indicators.csv')
    print('斑块指标计算完成，结果保存在{}'.format(output+'/patch_indicators.csv'))
           
def check_nichles_in_patch(patch_df,conditions1,conditions2,similarity):
    '''
    检查图斑范围内, conditions条件发生变化后, 在f_conditions是否存在, 返回存在个数和不存在单元个数
    patch_df: 包含conditions1,  conditions2名称数据框
    conditions1: 条件名称列表
    conditions2: 比较的条件名称列表
    similarity:  比较条件相似性
    '''
    patch_df_c = patch_df
    count_dis = 0
    count_exist = 0
    

    for index, row in patch_df.iterrows():
        #  取出现在条件对应的值
        for index, condition_name in enumerate(conditions1):
            # 找出对应的未来条件名称
            match_name = conditions2[index] if conditions2[index] else condition_name
            # 取出起始像元条件i的值
            condition_value = row[condition_name]

            if pd.isna(condition_value) or np.isinf(condition_value):
                patch_df_temp = []
            else:
                # 提取该变量的相似性条件
                cons_sim = similarity[index]

                if cons_sim == [0]:                                                     #  条件值相同
                    patch_df_temp = patch_df_c.loc[patch_df_c[match_name]==condition_value]
                        
                elif len(cons_sim) == 2:

                    if "-inf" in cons_sim:
                        patch_df_temp = patch_df_c[patch_df_c[condition_name] <= condition_value]
                    elif "inf" in cons_sim:
                        patch_df_temp = patch_df_c[patch_df_c[condition_name] >= condition_value]
                    else:
                        try:
                            con_list =list(range(int(cons_sim[0]+condition_value), int(cons_sim[1]+1+condition_value)))
                        except  Exception as e:
                            print(f"发生错误: {row['id'],cons_sim[0],condition_value,cons_sim[1]}")  
                            print(f"发生错误: {e}") 
                        patch_df_temp = patch_df_c.loc[patch_df_c[match_name].isin(con_list)]
                else:
                    con_list = cons_sim
                    patch_df_temp = patch_df_c.loc[patch_df_c[match_name].isin(con_list)]

            if len(patch_df_temp) > 0:
                patch_df_c = patch_df_temp
            else:
                count_dis += 1
                break

        if len(patch_df_temp) > 0:
            count_exist += 1
    # print(count_dis, count_exist,'/n' )
    # print(patch_df['patch_num'].values[0], len(patch_df), count_dis, count_dis/len(patch_df), count_exist, count_exist/len(patch_df) )
    return  count_dis, count_dis/len(patch_df), count_exist, count_exist/len(patch_df)      
    


if __name__ == '__main__':
    work_dir = '../example2'
    # sat_post_process(work_dir)
    sat_patch_indicator(work_dir, patch_name= 'patch_num', indicators=None)
    #sat_landscape_indicator(work_dir, indicators=None, time=100)

    
    