# -*- coding: utf-8 -*-
# 读写栅格

'''
def get_raster_information(rater_path,square =False,set_inf_criteria = None,set_inf_value = None):  ### '<'
    # rater_path: File path to a reference or resistance surface raster. 参考栅格或阻力面栅格的文件路径
    # square: Whether the resistance value needs to be squared, default False, Either False or True. 电阻值是否需要平方处理，默认为 False（假），False（假）或 True（真）
    # set_inf_criteria ：Whether or not to change the resistance surface for some values to infinity,default None, One of the five of '>','>=','==','<' and '<='.
    # set_inf_value: Specified values that need to be made infinite
    raster = rasterio.open(rater_path)
    raster_values = raster.read(1)
    rows, cols = raster_values.shape
    resolution = (raster.bounds.right - raster.bounds.left) / M
    if square:
        weight_list = [x*x for item in raster_values for x in item]  # 转换成一个列表   2023.0320更新 inx(x)改为x*x
    else:
        weight_list = [x for item in raster_values for x in item]  # 转换成一个列表
    if set_inf_criteria:
        weight_list =eval(" [float('inf') if i"+set_inf_criteria+str(set_inf_value)+" else i for i in weight_list]")  ### float('inf')
    return weight_list,rows, cols,resolution
'''
import rasterio
from osgeo import gdal
import numpy as np
import pandas as pd



def write_to_tiff(src_filename, dst_filename, bands_num, bands_arr, bandformat=gdal.GDT_Float32,
                  nodata=np.nan):  ####bandformat =  gdal.GDT_UInt16
    '''

    :param src_filename: 参考文件路径
    :param dst_filename: 保存的文件路径
    :param bands_num: 波段数
    :param bands_arr: 波段数据
    :param bandformat:
    :param nodata:
    :return:
    '''
    # 获取栅格信息
    src_ds = gdal.Open(src_filename)
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    geo = src_ds.GetGeoTransform()
    proj = src_ds.GetProjection()

    # im_bands = src_ds.RasterCount

    # 创建栅格
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(dst_filename, xsize, ysize, bands_num, bandformat)
    # outdata = driver.CreateCopy(dst_filename,src_filename,0)
    outDs.SetGeoTransform(geo)
    outDs.SetProjection(proj)  ####设定投影坐标系

    for i in range(bands_num):
        outband = outDs.GetRasterBand(i + 1)
        outband.WriteArray(bands_arr[i])
        outband.SetNoDataValue(nodata)  ######指定NODATA值
    outDs.FlushCache()
    outDs = None


# 根据栅格获取栅格信息,包括值列表,行数,列数 和分辨率
class ReadRaster:
    def __init__(
            self,
            file_dir=None,
            rows=None,
            cols=None,
            resolution=None,
    ):
        self.file_dir = file_dir
        self.raster = rasterio.open(self.file_dir)
        self.bands = self.raster.count
        raster_values = self.raster.read(1)
        self.rows, self.cols = raster_values.shape
        self.resolution = (self.raster.bounds.right - self.raster.bounds.left) / self.cols
        band_value = []
        for i in range(self.bands):
            value_lst = self.raster.read(i + 1)
            value_lst = [x for item in value_lst for x in item]  # 转换成一维列表
            band_value.append(value_lst)
        self.band_value_lst = band_value


def xy_from_cell(path):
    """
    根据栅格范围,栅格像元id 和 中心坐标id,x,y,
    :param path: # 栅格的路径
    :return: 由id,x,y 组成的数据框
    """

    ras = rasterio.open(path)
    raster_values = ras.read(1)
    rows, cols = raster_values.shape
    resolution = (ras.bounds.right - ras.bounds.left) / cols
    x, y = [], []
    for ii in range(0, rows):
        for jj in range(0, cols):
            coor = ras.xy(ii, jj)
            x.append(coor[0])
            y.append(coor[1])
            '''
            coor = ras.transform * (ii, jj)
            x.append(round(coor[0] + resolution / 2))
            y.append(round(coor[1] - resolution / 2))
            '''
    one = {'id': list(range(1, rows * cols + 1)), 'x': x, 'y': y}
    xy = pd.DataFrame(one)
    return xy
'''
work_dir = '../example/'
path_raster = work_dir+'conditions4.tif'

ra = ReadRaster(path_raster)
print(ra.cols, ra.rows, ra.resolution)
'''
