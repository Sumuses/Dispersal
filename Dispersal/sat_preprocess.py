# -*- coding: utf-8 -*-
# 将栅格信息整理进一个dataframe中,区分待求解像元和基础像元信息
# 最后输出Base_infor.csv, 和 Target_region.csv

from utils import mkdir,generate_configs_csv_json, get_raster_information, set_inf_in_list
from RW_raster import ReadRaster, xy_from_cell
import os
from creat_graph import get_nodes_and_edges, creat_graph
from loguru import logger
import json
import pandas as pd
import ast,re


def sat_preprocess(work_dir):

    '''
    将栅格数据集中在一个大型的csv表格中，并提取栅格的基本信息
    work_dir:工作目录
    '''
    generate_configs_csv_json(work_dir)

    with open(work_dir+'/configs.json','r', encoding='UTF-8') as f:
        configs = json.load(f)

    geo_dataset_list = pd.read_csv(configs['Geo_dataset_list'])
    geo_dataset_list.set_index("name", inplace=False) 
    geo_dataset_list= geo_dataset_list.sort_values(by='priority_for_niche')
    
    resistance_raster = geo_dataset_list[geo_dataset_list['name'] == configs['resistance_raster']]['path'].values[0]
    # print('阻力路径是{}'.format(resistance_raster))
    analyse_objects = configs['study_area']

    temp = mkdir(work_dir,'temp')        # 创建临时文件夹
    #temp = work_dir + '/temp'
    log = mkdir(temp, folder_name='log')
    #log = temp + '/log'

    logger.add(log +'//S1_raster_to_table{time}.log', encoding="utf-8", enqueue=True)
    txt = '【第1/6步】: 在"{0}"下创建"temp" 和 "log" 文件夹分别保存临时文件和日志'.format(work_dir)
    logger.info(txt)

    # First 检查以上文件是否存在于 work_dir 中
    txt = '【第2/6步】: 检查初始化字典中的文件名是否在work_dir:"{0}"有对应文件存在'.format(work_dir)
    print(txt)
    logger.info(txt)
    filename_list = os.listdir(work_dir)
    for path in geo_dataset_list['path']:
        if path[len(work_dir)+1:] in filename_list:
            ra = ReadRaster(path)
            if ra.bands == 1:
                pass
            else:
                txt = '中文：栅格"{0}"有{1}个波段,但条件命名列表中定义了1个,\n English: Raster "{0}" has {1} hands, but condition ' \
                    'naming list defines 1'.format(path, ra.bands)
                logger.warning(txt) 
        else:
            txt = '文件‘{0}’不存在于当前项目文件夹中, \n In English： {0} does not exist in the current project folder'.format(path)
            logger.warning(txt)
            print(path,path[len(work_dir)+1:],filename_list)
    txt = '【第3/6步】: 检查栅格文件的行列数，分辨率是否一致'
    logger.info(txt)
    row_lst, col_lst, resolution_lst, rastername= [], [], [], []
    for path in geo_dataset_list['path']:
        ra = ReadRaster(path)
        if ra.bands == 1:
            row_lst.append(ra.rows)
            col_lst.append(ra.cols)
            resolution_lst.append(ra.resolution)
            rastername.append(geo_dataset_list[geo_dataset_list['path']==path]['name'].values[0])
        else:
            txt = '中文：栅格"{0}"有{1}个波段,但条件命名列表中定义了1个,\n English: Raster "{0}" has {1} hands, but condition ' \
                    'naming list defines 1'.format(geo_dataset_list[geo_dataset_list['path']==path]['name'].values[0], ra.bands)
    if len(list(set(row_lst))) > 1 | len(list(set(col_lst))) > 1:   #  如果存在栅格行数/列数与其他的不一致
        txt = '【提示】存在栅格行数/列数与其他的不一致, \n name:{0} \n rows:{1} \n cols:{2}\n res:{2}'.format(rastername, row_lst, col_lst, resolution_lst)
        logger.error(txt)
    if len(list(set(resolution_lst))) > 1:
        txt = '【提示】存在栅格分辨率与其他的不一致, Please check the different raster resolution: res:{0}'.format(resolution_lst)
        logger.info(txt)

    # 创建数据框, 将每个栅格数据存在在表格中
    txt = '【第4/6步】: 创建数据框, 将每个栅格数据存于表格中'
    print(txt)
    logger.info(txt)

    xy = xy_from_cell(resistance_raster)
    for path in geo_dataset_list['path']:
        ra = ReadRaster(path)
        new_col = ra.band_value_lst[0]
        col_name = geo_dataset_list[geo_dataset_list['path']==path]['name'].values[0]
        xy[col_name] = new_col


    # 检查数据框各列是否为小数，如有，放大10倍
    txt = '【提示】【第5/6步】: 检查数据框各列是否有小数'
    logger.info(txt)
    zoom_index=[]
    for index, row in xy.items():
        zoom = False
        for value in xy[index]:
            if not value - int(value):    # 判断是否为整数
                pass
            elif index in ['id','x','y']: # 如果是id、x、y列，则不放大
                pass
            else:
                zoom = True
        if zoom:
            xy[index] = round(row * 10)
            txt = '{0}存在小数，将放大10倍，并取整'.format(index)
            zoom_index.append(index)
            logger.warning(txt)
    xy = xy.astype(int,errors='ignore')    # 

    # 保存数据和信息字典
    txt = '【第6/6步】: 将表格保存至临时文件夹"temp"中Base_infor.csv,Target_region.csv'
    logger.info(txt)

    # 全部信息保存至Base_infor.csv
    xy.to_csv(temp + '/Base_infor.csv', index=False, mode='w')


    # 筛选定义研究对象空间的区域
    analyse_objects    # 列表

    # 保存研究区域信息至Target_region.csv
    # Target_region = xy[xy['study_area'] > 0]
    Target_region = xy
    for name in analyse_objects:
        similarity = geo_dataset_list[geo_dataset_list['name'] == name]["similarity_for_variable"].values[0]
        if similarity != similarity:                                        # 判断是否为nan, nan自身不相等
            Target_region = Target_region[Target_region[name] > 0]
        else:
            if 'inf' in similarity:
                if '-' in similarity:
                    max = re.findall(r'-?\d+',  similarity) 
                    similarity = ["-inf",int(max[0])]
                else:
                    min = re.findall(r'-?\d+',  similarity) 
                    similarity = [int(min[0]),"inf"]
            else:
                similarity = ast.literal_eval(similarity)
            if len(similarity) == 2:
                Target_region = Target_region[(Target_region[name] >= float(similarity[0])) & (Target_region[name] <= float(similarity[1]))]
            else:
                Target_region = Target_region[Target_region[name].isin(similarity)]

    # Target_region = Target_region.drop(columns=f_conditions)

    # Target_region = Target_region.dropna()              # 去掉含有空值的行
    # Target_region1 = Target_region.copy()
    Target_region.to_csv(temp + '/Target_region.csv', index=False, mode='w')

    # 栅格行列信息
    rows, cols, resolution = row_lst[0], col_lst[0], resolution_lst[0]
    # 生态位条件信息
    conditions = geo_dataset_list[geo_dataset_list['is_niche_variable']==True]['name'].tolist()

    f_conditions = geo_dataset_list[geo_dataset_list['compare_to_future']!='FALSE']['compare_to_future'].tolist()
    c_conditions = [x for x in conditions if x not in set(f_conditions)]
    # 生态位未来条件名称
    compare_to_future = []
    for name in c_conditions:
        future_name = geo_dataset_list.loc[geo_dataset_list['name']==name,'compare_to_future'].values[0]
        if future_name!='FALSE':
            compare_to_future.append(future_name)
        else:
            compare_to_future.append(None)
            
    # 生态位条件相似性信息
    cons_similarity=[]
    essential_for_niche = []
    for name in c_conditions:
        similarity = geo_dataset_list[geo_dataset_list['name']==name]['similarity_for_variable'].values[0]
        if 'inf' in similarity:
            if '-' in similarity:
                max = re.findall(r'-?\d+',  similarity) 
                similarity = ["-inf",int(max[0])]
            else:
                min = re.findall(r'-?\d+',  similarity) 
                similarity = [int(min[0]),"inf"]
        else:
            similarity = ast.literal_eval(similarity)
        if name in zoom_index:
            similarity = [x * 10 if isinstance(x, (int, float)) else x for x in similarity]  # 如果该字段有小数，则乘以10，并跳过非数字元素
        cons_similarity.append(similarity)
        essential_for_niche.append(geo_dataset_list[geo_dataset_list['name']==name]['essential_for_niche'].values[0])


     # 获取栅格的基本信息：值列表, 行数, 列数
    rows, cols, resolution, weight_list = get_raster_information(rater_path=resistance_raster)

    # 设置无穷大
    weight_list_for_Graph = set_inf_in_list(lst=weight_list, set_inf_criteria='<', set_inf_value=1, square=True)

    # Finded_table = final_table #.copy(deep=True)
    G_edges = get_nodes_and_edges(neighbors=configs['neighbors'], rows=rows, cols=cols, edge_weight_list=weight_list_for_Graph)
    G, index_dict = creat_graph(tool="graph_tool", rows=rows, cols=cols, G_edges=G_edges)
    G.save(temp + "/my_graph.gt")


    infor = {
        'work_dir': work_dir,
        'temp': temp,
        'log': log,
        'rows': rows,
        'cols': cols,
        'resolution': resolution,
        'id_name': 'id',
        'conditions': c_conditions,
        'resistance_raster_path': resistance_raster,
        #'f_conditions': f_conditions,
        'compare_to_future': compare_to_future,
        'essential_for_niche': essential_for_niche,
        'cons_similarity':cons_similarity,
        'G':temp + "/my_graph.gt",
        'Gnode_index':index_dict
    }
    
    configs.update(infor)

    with open(work_dir+'/configs.json', 'w') as f_new:
        json.dump(configs, f_new)
        
    txt = '【完成】: 栅格信息已存在临时文件夹"temp"的Base_infor.csv,Target_region.csv中' \
    '\n 图形构建已完成'
    print(txt);logger.info(txt)

if __name__ == '__main__':
    work_dir = '../example'
    sat_preprocess(work_dir)
