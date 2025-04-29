# -*- coding: utf-8 -*-
import graph_tool as gt
from graph_tool.all import *
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from shapely import geometry
from osgeo import gdal
import os
import json
from shapelysmooth import taubin_smooth
from tqdm import tqdm
import geopandas as gpd
from keplergl import KeplerGl
import numpy as np


def sat_create(work_dir):
    '''
    生成shapefile格式的投影和非投影的sats
    '''
    print("第1/4步，汇总路径基本数据")
    temp = work_dir + '/temp'
    output = work_dir + '/output'
    with open(work_dir  + "/configs.json", "r") as f:  # 打开文件
        configs = json.load(f)  # 读取文件

    # 读取基础表格
    base_infor = pd.read_csv(temp + "/base_infor.csv")
    LCPs_index_name = []
    LCPs_nodes_name = []
    filename_list = os.listdir(temp)
    for filename in filename_list:
        if filename[:10] == 'LCPs_index':  # 将读取文件名字的前十个字符与'LCPs_index'匹对
            LCPs_index_name.append(filename)
        elif filename[:9] == 'LCPs_list':  # 将读取文件名字的前9个字符与'LCPs_list'匹对
            LCPs_nodes_name.append(filename)
        else:
            pass

    LCPs_list = []
    for file in LCPs_nodes_name:
        with open(temp + '/' +file, "r") as f:  # 打开文件
            data = json.load(f)  # 读取文件
        LCPs_list = LCPs_list + data

    LCPs_index = []
    for file in LCPs_index_name:
        with open(temp + '/' + file, "r") as f:  # 打开文件
            data = json.load(f)  # 读取文件
        LCPs_index = LCPs_index + data

    print("第2/4步，获取栅格中的投影信息")
    src_ds = gdal.Open(configs['resistance_raster_path'])
    proj = src_ds.GetProjection()

    print("第3/4步，整理路径的节点信息，并平滑处理")
    geo_Series = []
    i = 0
    for index, lst in tqdm(enumerate(LCPs_list),desc="正在进行平滑处理,共{0}项".format(len(LCPs_list))):       #Bar('正在进行平滑处理').iter(LCPs_list):   # tqdm(range(n,m),desc="正在分析",ncols=80) 
    # print("正在进行平滑处理: {:.0%}".format(index+1 / len(LCPs_list)))
        one_line_points = []
        if len(lst) > 1:
            for point in lst:
                x = base_infor[base_infor["id"] == point].x.tolist()[0]
                y = base_infor[base_infor["id"] == point].y.tolist()[0]
                xy = (x, y)
                #print(xy)
                one_line_points.append(xy)
            geo_LineString = geometry.LineString(one_line_points)
            # 平滑曲线
            if len(one_line_points) > 10:
                steps = 5
                geo_LineString = taubin_smooth(geo_LineString, factor=0.5, mu=0.5, steps=steps)
            elif len(one_line_points) == 0:
                pass
            else:
                steps = 5
                geo_LineString = taubin_smooth(geo_LineString, factor=0.5, mu=0.5, steps=steps)

            geo_Series.append(geo_LineString)
        else:
            del LCPs_index[index-i]
            i +=1

    print("第4/4步，将路径写入shapefile文件")
    lines = gpd.GeoSeries(geo_Series,
                             crs=proj,
                             index=LCPs_index)

    lines.to_file(output + '/sats.shp',
               driver='ESRI Shapefile',
               encoding='utf-8')
    print("迁徙路径提取完成，shp格式保存在工作目录output文件夹下")

    # 读取 Shapefile
    shapefile_path = output + '/sats.shp'
    gdf = gpd.read_file(shapefile_path)

    # 检查当前坐标系
    print(f"当前坐标系为: {gdf.crs}")

    # 重新投影到 WGS84 (EPSG:4326)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    # 保存新的 Shapefile
    output_path = output + '/sats_noproj.shp'
    gdf_wgs84.to_file(output_path, driver="ESRI Shapefile")

    print(f"坐标转换已完成，当前坐标为WGS84地理坐标系，新文件已保存至: {output_path}")


def shpsats_to_geojson(work_dir, start_time, end_time ):
    output = work_dir + '/output'
    # 读取 Shapefile 文件
    gdf = gpd.read_file(output + '/sats_noproj.shp')

    # 确保数据包含几何信息
    if gdf.geometry.is_empty.any():
        raise ValueError("Shapefile 中包含空几何，请检查数据！")

    # 设定固定的起始时间和结束时间
    start_time = pd.Timestamp(start_time)  # 轨迹开始时间
    end_time = pd.Timestamp(end_time)    # 轨迹结束时间

    # 生成 GeoJSON 结构
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    geojson_data_ontime = {
        "type": "FeatureCollection",
        "features": []
    }

    # 遍历每条轨迹
    for idx, row in gdf.iterrows():
        line = row.geometry  # 获取 LineString
        num_points = len(line.coords)  # 获取轨迹点数
        if num_points < 2:
            continue  # 如果轨迹点不足，跳过

        # 计算均匀分布的时间戳
        timestamps = np.linspace(start_time.timestamp(), end_time.timestamp(), num_points)

        # 处理坐标数据，添加时间戳（转换为 Unix 时间戳）
        coordinates = [
            [lon, lat, 0, int(timestamps[i])]  
            for i, (lon, lat) in enumerate(line.coords)
        ]

        # 处理坐标数据（去除时间戳，仅保留 [lon, lat]）
        coordinates_notime = [[lon, lat] for lon, lat in line.coords]


        # 构造 Feature
        feature = {
            "type": "Feature",
            "properties": {
                "vendor": "A"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }
        
        geojson_data["features"].append(feature)

        # 构造 Feature
        feature_notime = {
            "type": "Feature",
            "properties": row.drop("geometry").to_dict(),  # 保留所有属性字段
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates_notime
            }
        }
        
        geojson_data_ontime["features"].append(feature_notime)

    # 保存为 GeoJSON 文件
    with open(output + "/sats_timeseries.geojson", "w") as f:
        json.dump(geojson_data, f, indent=2)

    with open(output + "/sats_notimeseries.geojson", "w") as f:
        json.dump(geojson_data_ontime, f, indent=2)

    print("GeoJSON 文件已生成:sats_timeseries.geojson")


def sat_visualize(workdir, start_time='2025-01-01 00:00:00', end_time = "2025-03-11 00:00:00"):
    '''

    '''
    output = work_dir + '/output'
    # 创建 Kepler.gl 地图对象
    map_ = KeplerGl()

    
    with open('../example2/output/sats_timeseries.geojson', 'r') as f:
        geojson = f.read()

    with open('../example2/output/sats_notimeseries.geojson', 'r') as f:
        geojson_notime = f.read()

    # 添加数据集
    map_.add_data(data=geojson, name="trip_layer")
    map_.add_data(data=geojson_notime, name="line_layer")

    # 定义 `trip` 图层配置
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                    "id": "trip_layer",
                    "type": "trip",
                    "config": {
                        "dataId": "trip_layer",
                        "label": "Migration Animation",
                        "color": [255, 0, 0],
                        "columns": {
                            "geojson": None,
                            "lat": "latitude",
                            "lng": "longitude",
                            "altitude": None,
                            "timestamp": "time"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "trailLength": 3000,  # 轨迹持续时间
                            "colorRange": {
                                "colors": ["#FF0000", "#00FF00", "#0000FF"]
                            },
                            "thickness": 20
                        }
                    }
                },{
                        "id": "line_layer",
                        "type": "line",
                        "config": {
                            "dataId": "Line Layer",
                            "isVisible": True,
                            "color": [0, 0, 255],  # 蓝色线路
                            "opacity": 0.01,  # 设置透明度 1%
                            "strokeWidth": 2
                        }
                    }
                ],
                "animationConfig": {
                    "currentTime": None,  # 关闭时间播放条
                    "speed": 1  # 播放速度
                }
            }
        }
    }

    # 应用配置
    map_.config = config

    # 保存为本地 HTML 文件
    map_.save_to_html(file_name= output + '/sats_map.html')


if __name__ == '__main__':
    work_dir = '../example2'
    # sat_create(work_dir)
    sat_visualize(work_dir)

    


