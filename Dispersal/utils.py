import numpy as np
import math
import os
import rasterio
import pandas as pd
import os
from osgeo import ogr

from progress.bar import Bar
import geopandas as gpd
from tqdm import tqdm
from rasterio.warp import transform
import xarray as xr
from pyproj import Proj, Transformer

def lcp_length(path_list, resolution=None, rows=None, cols=None):
    """
    根据路径节点计算LCP长度，主要在8领域时候使用
    :param path_list : 路径节点列表
    :param resolution ： 栅格像元的分辨率
    :param rows ： 栅格的行数
    :param cols ： 栅格的列数
    """
    # 计算两个相邻节点之间的差值的绝对值
    diff_path = np.absolute(np.diff(path_list))
    # 统计差值为1或cols的数量
    single_length = np.sum((diff_path == 1) | (diff_path == cols))
    # 根据斜边数 和 直边数 求总长度
    length = ((len(diff_path) - single_length) * 1.414 + single_length) * resolution
    return length


def get_raster_information(rater_path):
    """
    获取地理栅格的行数、列数、分辨率 和 值的一维列表
    :param rater_path: File path to a reference or resistance surface raster.
    栅格或阻力面栅格的文件路径
    """
    raster = rasterio.open(rater_path)
    raster_values = raster.read(1)
    rows, cols = raster_values.shape
    resolution = (raster.bounds.right - raster.bounds.left) / cols
    weight_list = [x for item in raster_values for x in item]  # 转换成一个列表
    return rows, cols, resolution, weight_list


def set_inf_in_list(lst,
                    set_inf_criteria=None,
                    set_inf_value=None,
                    square=False):
    """
    修改某些阻力面的值
    :param lst : 一维数组 list
    :param set_inf_criteria ：Whether or not to change the resistance surface for some values to infinity,
    default None, One of the five of '>','>=','==','<' and '<='.
    :param set_inf_value: Specified values that need to be made infinite

    :param square: Whether the resistance value needs to be squared, default False, Either False or True.
    (栅格)电阻值是否需要平方处理，默认为 False（假），False（假）或 True（真）
    """
    weight_list = lst
    # 将部分值更改为无线大
    if set_inf_criteria:
        weight_list = eval(
            " [float('inf') if i" + set_inf_criteria + str(set_inf_value) + " else i for i in weight_list]")

    if square:
        weight_list = [x * x for x in weight_list]  # 转换成一个列表, 并将数值平方

    return weight_list


Release_list_element = lambda x: [y for l in x for y in Release_list_element(l)] if type(x) is list else [x]  # 将多层列表展开为一个列表
"""
将多层嵌套的列表，转换成一维列表
"""


def mkdir(work_dir, folder_name):
    """
    创建临时文件夹
    :param work_dir ： 创建文件夹的保存路径
    :param folder_name ： 创建文件夹名字
    """
    path = work_dir +'/'+ folder_name
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        return path
    else:
        return path


def azimuthangle(x1, y1, x2, y2):
    """
    已知两点坐标计算角度 -
    :param x1: 原点横坐标值
    :param y1: 原点纵坐标值
    :param x2: 目标点横坐标值
    :param y2: 目标纵坐标值
    :return: 返回径向角度 0-360°
    """
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return angle * 180 / math.pi


def count_real_sum_cost(node_lst, ori_cost_lst):
    """
    根据节点列表求路径在原阻力面栅格上的累积阻力
    :param node_lst: 路径节点编号
    :param ori_cost_lst: 原阻力面的一维数组
    :return: 返回原始阻力面下的累积阻力值
    """
    node_lst = [x - 1 for x in node_lst]
    ori_cost_lst = np.array(ori_cost_lst)
    cost_list_follow_nodes = ori_cost_lst.take(node_lst)
    return cost_list_follow_nodes.sum()


def isexist(name, path=None):
    """
    # 检查某路径下，文件是否存在
    :param name: 需要检测的文件或文件夹名
    :param path: 需要检测的文件或文件夹所在的路径，当path=None时默认使用当前路径检测
    :return: True/False 当检测的文件或文件夹所在的路径下有目标文件或文件夹时返回Ture,
            当检测的文件或文件夹所在的路径下没有有目标文件或文件夹时返回False
    """
    if path is None:
        path = os.getcwd()
    if os.path.exists(path + '/' + name):
        print("Under the path: " + path + '\n' + name + " is exist")
        return True
    else:
        if os.path.exists(path):
            print("Under the path: " + path + '\n' + name + " is not exist, ")
        else:
            print("This path could not be found: " + path + '\n')
        return False


def tif2nc(tif_file, nc_file):
    """
    将tif栅格数据转成nc数据格式
    :param tif_file: 输入文件路径
    :param nc_file: 输出文件路径
    :return: 保存nc文件至本地
    """
    from osgeo import gdal
    # 打开tif文件
    tif_dataset = gdal.Open(tif_file)
    if tif_dataset is None:
        print("无法打开tif文件！")
        exit(1)

    # 获取tif文件的相关信息
    width = tif_dataset.RasterXSize
    height = tif_dataset.RasterYSize
    band_count = tif_dataset.RasterCount

    # 创建nc文件
    driver = gdal.GetDriverByName("netCDF")
    nc_dataset = driver.Create(nc_file, width, height, band_count, gdal.GDT_Float32)
    if nc_dataset is None:
        print("无法创建nc文件！")
        exit(1)

    # 将tif文件的信息写入nc文件
    for i in range(1, band_count + 1):
        band = tif_dataset.GetRasterBand(i)
        data = band.ReadAsArray()
        nc_dataset.GetRasterBand(i).WriteArray(data)

    # 设置nc文件的坐标系统和投影信息
    nc_dataset.SetProjection(tif_dataset.GetProjection())
    nc_dataset.SetGeoTransform(tif_dataset.GetGeoTransform())

    # 关闭文件
    tif_dataset = None
    nc_dataset = None

    print("转换完成！")


def raster2_84(src_img, dst_img):
    """
    将栅格转换成WGS84地理坐标系
    :param src_img: 输入栅格
    :param dst_img: 输出栅格
    :return:
    """
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio import crs

    # 转为地理坐标系WGS84
    dst_crs = crs.CRS.from_epsg('4326')

    with rasterio.open(src_img) as src_ds:
        profile = src_ds.profile

        # 计算在新空间参考系下的仿射变换参数，图像尺寸
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_ds.crs, dst_crs, src_ds.width, src_ds.height, *src_ds.bounds)

        # 更新数据集的元数据信息
        profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': 0
        })

        # 重投影并写入数据
        with rasterio.open(dst_img, 'w+', **profile) as dst_ds:
            for i in range(1, src_ds.count + 1):
                src_array = src_ds.read(i)
                dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype'])

                reproject(
                    # 源文件参数
                    source=src_array,
                    src_crs=src_ds.crs,
                    src_transform=src_ds.transform,
                    # 目标文件参数
                    destination=dst_array,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    # 其它配置
                    resampling=Resampling.cubic,
                    num_threads=2)

                dst_ds.write(dst_array, i)


def create_fishnet(input_vector_file, output_grid_file, grid_size):
    """
    #创建shp渔网
    :param input_vector_file: 输入shp文件
    :param output_grid_file: 输出渔网文件
    :param grid_size: 像元大小
    :return: 输出渔网文件
    """
    # 打开矢量文件
    input_source = ogr.Open(input_vector_file)
    input_layer = input_source.GetLayer()

    # 获取矢量文件的空间参考
    srs = input_layer.GetSpatialRef()

    # 获取矢量文件的范围
    x_min, x_max, y_min, y_max = input_layer.GetExtent()

    # 计算渔网的行列数
    rows = int(round((y_max - y_min) / grid_size)) + 1
    cols = int(round((x_max - x_min) / grid_size)) + 1
    print("即将创建行为{}，列为{},共{}个单元的格网".format(rows, cols, rows * cols))
    # print("四至x_min{},x_max{}, y_min{}, y_max{}".format(x_min, x_max, y_min, y_max))

    # 创建输出文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_grid_file):
        driver.DeleteDataSource(output_grid_file)
    output_source = driver.CreateDataSource(output_grid_file)
    output_layer = output_source.CreateLayer(output_grid_file, srs, ogr.wkbPolygon)

    # 创建ID字段
    id_field = ogr.FieldDefn("id", ogr.OFTInteger)
    output_layer.CreateField(id_field)

    # 创建网格
    id = 0
    y = y_min
    with tqdm(total=rows,ncols=80) as pbar:
    #bar = Bar('正在创建渔网',max =rows )
        while y < y_max:
            x = x_min
            while x < x_max:
                # print(f"x = {x}")
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(x, y)
                ring.AddPoint(x + grid_size, y)
                ring.AddPoint(x + grid_size, y + grid_size)
                ring.AddPoint(x, y + grid_size)
                ring.AddPoint(x, y)
                ring.CloseRings()

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                feature = ogr.Feature(output_layer.GetLayerDefn())
                feature.SetField("id", id)
                feature.SetGeometry(poly)
                output_layer.CreateFeature(feature)

                feature = ring = poly = None
                id += 1
                x += grid_size
            y += grid_size
            pbar.update(1)
        #bar.next()
        # print("正在创建渔网: {:.0%}".format(y/y_max))
    # 循环完成后调用finish()方法
    # bar.finish()
    input_source = output_source = None
    


def average_angle(geo):
    """
    计算每个格网内线的总方向
    :param geo:地理数据geodataframe格式
    :return:
    """
    # 初始化列表来存储每条折线的走向
    headings = []

    # 遍历GeoDataFrame中的每条折线
    for line in geo['geometry']:
        # 对于折线，我们取第一个线段的方向作为折线的方向
        # 第一个线段的起点和终点分别是折线的起点和第二个点
        start_point = line.coords[0]  # 起点
        end_point = line.coords[-1]  # 第一个线段的终点（也是折线的“方向性终点”）

        # 计算第一个线段的向量
        vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])

        # 计算方向（与x轴的夹角，范围在-pi到pi之间）
        heading = np.arctan2(vector[1], vector[0])
        headings.append(heading)

    # 由于方向是循环的（即0和2pi是相同的），故不能直接对它们求平均
    # 一种方法是使用向量的平均方向来计算平均方向
    # 这里我们直接使用headings列表中的值来计算
    average_vector = np.mean(np.array(headings)[:, None] * np.array([np.cos(headings), np.sin(headings)]).T, axis=0)
    average_heading = np.arctan2(average_vector[1], average_vector[0])

    # 将平均方向从弧度转换为度数（0到360度之间）
    average_heading_deg = np.degrees(average_heading)
    # 由于arctan2的结果在-pi到pi之间，转换后可能在-180到180之间，我们需要确保它在0到360之间
    if average_heading_deg < 0:
        average_heading_deg += 360
    # 此时的角度为x轴为0,y轴为90度的坐标系下的角度，我们需要将其转换为x轴为90,y轴为0度的坐标系下的角度。即正北为0度。

    if average_heading_deg <= 90:
        average_heading_deg = abs(average_heading_deg-90)
    elif average_heading_deg <= 180:
        average_heading_deg = 450-average_heading_deg
    elif average_heading_deg <= 270:
        average_heading_deg = 450-average_heading_deg
    elif average_heading_deg <= 360:
        average_heading_deg = 450 - average_heading_deg
    else:
        pass
    # 返回平均方向
    return average_heading_deg



# GEOTIFF 转 NCTIF
def read_tif(tif_path):
    """
    读取tif文件
    :param tif_path: tif文件路径
    :return: tif文件数据
    """
    with rasterio.open(tif_path) as dataset:
        crs = dataset.crs  # 获取投影信息
        transform_matrix = dataset.transform  # 获取仿射变换矩阵
        data = dataset.read()  # 读取所有波段数据
        speed = data[0]
        angle = data[1]
        #
        data[0]= speed * np.cos(angle)
        data[1]= speed * np.sin(angle)

        width, height = dataset.width, dataset.height
        
        # 生成栅格的网格点坐标
        x_coords, y_coords = np.meshgrid(
            np.arange(width), np.arange(height)
        )
        x_coords = transform_matrix * (x_coords, y_coords)
        return data, x_coords[0], x_coords[1], crs

# 进行坐标转换
def convert_to_geographic(x, y, src_crs):
    """
    将坐标从 src_crs 转换为地理坐标系 (EPSG:4326)
    :param x: x坐标
    :param y: y坐标
    :param src_crs: 源坐标系
    """
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

# 创建 NetCDF 文件
def create_nc(lon, lat, u_wind, v_wind, nc_path):
    """
    将 u_wind 和 v_wind 数据写入 NetCDF 文件
    :param lon: 经度
    :param lat: 纬度
    :param u_wind: u 风速
    :param v_wind: v 风速
    :param nc_path: NetCDF 文件路径
    """
    ds = xr.Dataset(
        {
            "u_wind": (("lat", "lon"), u_wind),
            "v_wind": (("lat", "lon"), v_wind),
        },
        coords={"lat": lat[:, 0], "lon": lon[0, :]},
    )
    ds.to_netcdf(nc_path)

# 主函数
def tif_to_nc(tif_path, nc_path):
    """
    将带有投影坐标系的 GeoTIFF 文件(速度，方向)转换为 NetCDF 文件(地理坐标系,uv速度分量)
    :param tif_path: GeoTIFF 文件路径
    :param nc_path: NetCDF 文件路径

    """
    noprj_tif = tif_path[-3]+''+'noproj.tif'
    raster2_84(tif_path, noprj_tif)
    data, lon, lat, crs = read_tif(noprj_tif)
    #lon, lat = convert_to_geographic(x, y, crs)
    u_wind, v_wind = data[0], data[1]           # 假设第1个波段是 速度，第2个是 方向角度
    create_nc(lon, lat, u_wind, v_wind, nc_path)

'''
def remove_proj_toWGS84(input_shapefile_path, output_shapefile_path):
    """
    去掉shapefile的投影信息，使其转化成WGS84地理坐标系

    """

    # 读取 Shapefile
    input_shapefile_path = "/mnt/c/users/su/onedrive - 南京大学/displace/example2/output/Displace_LCPs.shp"
    gdf = gpd.read_file(input_shapefile_path)

    # 检查当前坐标系
    print(f"原始坐标系: {gdf.crs}")

    # 重新投影到 WGS84 (EPSG:4326)
    gdf_wgs84 = gdf.to_crs(epsg=4326)

    # 保存新的 Shapefile
    output_shapefile_path = "../example2/output/Displace_LCPs_noproj.shp"
    gdf_wgs84.to_file(output_shapefile_path, driver="ESRI Shapefile")

    print(f"投影信息去除完成，只保留WGS84地理坐标信息，新文件已保存至: {output_shapefile_path}")
'''
import csv, json,sys
def generate_configs_csv_json(work_dir):
    """Generate a CSV file of raster information"""
    CSV_FILE = work_dir+'/geo_dataset_list.csv'
    header = ["name", "path", "is_niche_variable", "compare_to_future", "similarity_for_variable", "priority_for_niche", "essential_for_niche"]
    data = [
        ["example1", "../example/study_area.tif",False,False],
        ["example2", "../example2/resistance.tif",True,False,  ["-inf", 0],5,False],
        ["example3", "../example/landuse.tif",True,False,[21, 22, 23,24,31,32,33],1,True],
        ["example4", "../example/current_PCA1.tif",True,"example7", [-4, 4],2,True],
        ["example5", "../example/current_PCA2.tif",True,"example8",[-8, 8],3,False],
        ["example6", "../example/current_PCA3.tif",True,"example9",[-12, 12],4,True],
        ["example7", "../example/future_PCA1.tif",True,False],
        ["example8", "../example/future_PCA2.tif",True,False],
        ["example9", "../example/future_PCA3.tif",True,False]
    ]
    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)  # 写入数据
        print(f"geo_dataset_list.csv generated successfully, it is saved to'{CSV_FILE}' ")
    else:
        print(f"geo_dataset_list.csv already existed, please check'{CSV_FILE}' ")
    
    # generate configs for simulate adaptive trajectory
    JSON_FILE = work_dir+'/configs.json'
    content = {
        "work_dir": work_dir,
        "Geo_dataset_list": CSV_FILE,
        "study_area": "example_path",
        "resistance_raster": "example_path",
        "neighbors": 8,
        "num_of_optimal_SATs": 1
    }

    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, mode='w') as file:
            json.dump(content, file, indent=4)
        print(f"configs.JSON generated successfully, it is saved to '{JSON_FILE}' ")
        sys.exit("请参照操作文档，编辑修改geo_dataset_list.csv 和 configs.json 文件内容")
    else:
        print(f"configs.JSON already existed, please check '{JSON_FILE}'")