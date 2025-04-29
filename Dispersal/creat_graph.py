# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:54:07 2023
@author: Su,Jie
Email: sujienju@163.com
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
from progress.bar import Bar

def get_nodes_and_edges(neighbors=8, rows=None, cols=None, edge_weight_list=None):
    if rows * cols > len(edge_weight_list):
        print('The number of rows and columns should be consistent with the resistance surface')
    else:
        # matrices = np.linspace(1, m * n, m * n).reshape([n, m])  ####创建一个N行M列的数组
        # matrices = matrices.astype(int)  ###数组设置为整数
        list1 = []
        if neighbors == 4:
            for i in tqdm(range(1, cols * rows),desc="正在获取栅格邻接信息",ncols=80): # Bar('正在获取栅格邻接信息').iter(range(1, cols * rows)):
                if cols * rows - i < cols:  ####最后一行
                    res = (edge_weight_list[i - 1] + edge_weight_list[i]) / 2
                    if res == float('inf'):
                        pass
                    else:
                        one_edge = (str(i), str(i + 1), res)
                        list1.append(one_edge)

                elif i % cols != 0:  ####非最后一行
                    res = (edge_weight_list[i - 1] + edge_weight_list[i]) / 2
                    if res == float('inf'):
                        pass
                    else:
                        one_edge = (str(i), str(i + 1), res)
                        list1.append(one_edge)
                    res = (edge_weight_list[i - 1] + edge_weight_list[i + cols - 1]) / 2
                    if res == float('inf'):
                        pass
                    else:
                        one_edge = (str(i), str(i + cols), res)
                        list1.append(one_edge)
                else:  ####最后一列
                    res = (edge_weight_list[i - 1] + edge_weight_list[i + cols - 1]) / 2
                    if res == float('inf'):
                        pass
                    else:
                        one_edge = (str(i), str(i + cols), res)
                        list1.append(one_edge)

        elif neighbors == 8:
            for i in tqdm(range(0, cols * rows-1),desc="正在获取栅格邻接信息",ncols=80): #  Bar('正在获取栅格邻接信息').iter(range(0, cols * rows-1)):
                if edge_weight_list[i] == float('inf'):
                    continue
                if i < cols - 1:  ####  第一行
                    if edge_weight_list[i] != float('inf'):
                        res = (edge_weight_list[i] + edge_weight_list[i + 1]) / 2  ###右连接
                        one_edge = (str(i), str(i + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i + cols + 1] != float('inf'):  # 右下连接
                        res = (edge_weight_list[i] + edge_weight_list[i + cols + 1]) / 2
                        one_edge = (str(i), str(i + cols + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i + cols] != float('inf'):  ####下连接
                        res = (edge_weight_list[i] + edge_weight_list[i + cols]) / 2
                        one_edge = (str(i), str(i + cols), res)
                        list1.append(one_edge)
                    else:
                        pass

                elif (i + 1) % cols == 0:  ####  最后一列
                    if edge_weight_list[i + cols] != float('inf'):  ####下连接
                        res = (edge_weight_list[i] + edge_weight_list[i + cols]) / 2
                        one_edge = (str(i), str(i + cols), res)
                        list1.append(one_edge)
                    else:
                        pass

                elif cols * rows - i - 1 < cols:  ####最后一行,根据数组，即 i 不等于m*n
                    if edge_weight_list[i + 1] != float('inf'):
                        res = (edge_weight_list[i] + edge_weight_list[i + 1]) / 2  ###右连接
                        one_edge = (str(i), str(i + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i - cols + 1] != float('inf'):
                        res = (edge_weight_list[i] + edge_weight_list[i - cols + 1]) / 2  ###右上连接
                        one_edge = (str(i), str(i - cols + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                else:
                    if edge_weight_list[i - cols + 1] != float('inf'):
                        res = (edge_weight_list[i] + edge_weight_list[i - cols + 1]) / 2  ###右上连接
                        one_edge = (str(i), str(i - cols + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i + 1] != float('inf'):
                        res = (edge_weight_list[i] + edge_weight_list[i + 1]) / 2  ###右连接
                        one_edge = (str(i), str(i + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i + cols + 1] != float('inf'):  ### 右下连接
                        res = (edge_weight_list[i] + edge_weight_list[i + cols + 1]) / 2
                        one_edge = (str(i), str(i + cols + 1), res)
                        list1.append(one_edge)
                    else:
                        pass

                    if edge_weight_list[i + cols] != float('inf'):  ####下连接
                        res = (edge_weight_list[i] + edge_weight_list[i + cols]) / 2
                        one_edge = (str(i), str(i + cols), res)
                        list1.append(one_edge)
                    else:
                        pass

        else:
            print("The neighbors must be one of 4 or 8.")
        return list1
def creat_graph(tool = "graph_tool",rows = None, cols = None, G_edges = None):
    '''
    构建graph
    '''

    if tool == "graph_tool":
        import graph_tool as gt
        G = gt.Graph(G_edges, hashed=True, eprops=[('weight', 'double')])
        G.set_directed(False)        #####将栅格像元编号与图的节点的index对应

        index_dict = {}
        for i in tqdm(range(0, rows*cols),desc="正在为阻力面网络赋值",ncols=80):  # Bar('正在为阻力面网络赋值').iter(range(0, rows*cols)):  ###index from 0 to m*n-1
            index_dict.update({G.vp.ids[i]: i})
            # print("正在: {:.0%}".format((i+1) / rows*cols))
        '''
        inversion_index_dict = {v : k for k, v in index_dict.items()}    ###键，值互换
        '''
        return G, index_dict
    '''
    if tool == "igraph":
        import igraph as ig
        #节点编号从0开始，所有减1
        cudf_frame = pd.DataFrame(G_edges, columns=('source', 'destination', 'edge_attr'))
        cudf_frame['source'] = cudf_frame['source'].astype('int')
        cudf_frame['destination'] = cudf_frame['destination'].astype('int')
        cudf_frame['source'] = cudf_frame['source'] -1
        cudf_frame['destination'] = cudf_frame['destination']-1
        G = ig.Graph.DataFrame(cudf_frame, directed=False)
        return G
    if  tool == "cugraph":
        import cugraph
        cudf_frame = pd.DataFrame(G_edges, columns=('source', 'destination', 'edge_attr'))
        cudf_frame['source'] = cudf_frame['source'].astype('int')
        cudf_frame['destination'] = cudf_frame['destination'].astype('int')
        G = cugraph.Graph()
        G.from_pandas_edgelist(cudf_frame, source='source', destination='destination',edge_attr='edge_attr', renumber=True)
        return G
    '''