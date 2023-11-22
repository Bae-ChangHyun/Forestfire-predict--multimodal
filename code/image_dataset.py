import os
import numpy as np
import pandas as pd
from itertools import product

import natsort
from tqdm import tqdm
from glob import glob

import cv2
import matplotlib.pyplot as plt

import rasterio
from osgeo import gdal
import geopandas as gpd
from pyproj import Transformer

from load_variables import load_env
api_key,root_path,db_path,image_size=load_env()

"""
래스터파일과 shp형식으로 되어 있는 지형파일을 400*278의 격자로 나눠 잘라서 데이터 셋을 생성    
"""

# 원하는 사이즈로 특정 좌표를 중심으로 하는 이미지 crop
def crop_images(InputImage, OutputImage, latitude, longitude, CropSize):
    
    # 강원도 경계를 나타내는 래스터 파일
    RefImage = f'{root_path}/data/geo_data/gw_boundary/boundary_blank_resized.tif'
    # 참조 이미지의 boundary를 가져옴 
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)
    width = Image.RasterXSize
    height = Image.RasterYSize
    Image = None

    rds = rasterio.open(RefImage)
    left, bottom, right, top = rds.bounds
    resolution_x = (right - left) / width
    resolution_y = (top - bottom) / height

    transformer = Transformer.from_crs( 'EPSG:4326', 'EPSG:4326')
    longitude, latitude = transformer.transform(longitude, latitude)
    
    left_box = latitude - (resolution_x * CropSize)
    top_box = longitude + (resolution_y * CropSize)
    right_box = latitude + (resolution_x * CropSize)
    bottom_box = longitude - (resolution_y * CropSize)
    window = (left_box, top_box, right_box, bottom_box)

    gdal.Translate(OutputImage, InputImage, projWin = window)
    
# ndvi 년.월별 crop function
def ndvi_filtering(o_data,crop_size,filepath):
    
    for i in tqdm(range(len(o_data))):
        
        filename=str(o_data['date'][i])[:-2]
        
        year = filename[:4]
        month = int(filename[4:])
    
        if 3 <= month <= 5: month = 4 
        elif 6 <= month <= 9:month = 8
        elif 10 <= month <= 11: month = 11
        else: month = 1
        
        filename=year+str(month).zfill(2)
        
        InputImage = f'{root_path}/data/geo_data/raw/NDVI/{filename}.tif' 
        OutputImage = filepath+'/Crop_NDVI_'+str(i)+'.tif'
        
        lon=o_data['lon'][i]
        lat=o_data['lat'][i]
        
        crop_images(InputImage, OutputImage, lon, lat, crop_size)
        
def crop_train(data_n,o_data,crop_size):
    filepath=f"{root_path}/data/modeling/image_data/crop/train/{data_n}"
    os.makedirs(filepath, exist_ok=True)
    
    print(f"Start get {data_n} information")
    tmp=natsort.natsorted(glob(filepath+"/*.tif"))
    
    if(len(tmp)==len(o_data)):
        print(f"---> {data_n} train crop image already existed")
        return
    else: print(f"Create {data_n} train crop image start")

    # ndvi의 경우 년,월별로 데이터가 다르기 때문에 따로 작업을 수행하여야 한다.
    if(data_n=="NDVI"):
        ndvi_filtering(o_data,1,filepath)
        print("--> Complete")
        return 
    
    InputImage = f'{root_path}/data/geo_data/raw/{data_n}_gw.tif'   
    
    for i in tqdm(range(len(o_data))):
        OutputImage = filepath+'/Crop_'+data_n+'_'+str(i)+'.tif'
        
        lon=o_data['lon'][i]
        lat=o_data['lat'][i]
        
        crop_images(InputImage, OutputImage, lon, lat, crop_size)
    print(f"Complete")
    
def crop_test(data_n,crop_size,width_num,height_num):
    
    filepath=f"{root_path}/data/modeling/image_data/crop/test/{data_n}"
    os.makedirs(filepath, exist_ok=True)
    
    print('#'*20)
    print(f"Start get {data_n} information")
    tmp=natsort.natsorted(glob(filepath+"/*.tif"))
    
    if(len(tmp)==111200):
        print(f"---> {data_n} test crop image already existed")
        return
    else: print(f"Create {data_n} test crop image start")
    
    if(data_n=='NDVI'):InputImage = f'{root_path}/data/geo_data/raw/NDVI/202204.tif'   
    else:InputImage = f'{root_path}/data/geo_data/raw/{data_n}_gw.tif'   

    coordinates_combinations = product(height_num, width_num)
    for num, (lat, lon) in tqdm(enumerate(coordinates_combinations), total=len(height_num) * len(width_num)):
        OutputImage = f"{filepath}/Crop_{data_n}_{num}.tif"
        crop_images(InputImage, OutputImage, lon, lat, crop_size)
    
    print("--> Complete")
    
def convert_npy(data_n,types):
    filepath=f"{root_path}/data/modeling/image_data/{image_size}/{types}/"
    os.makedirs(filepath, exist_ok=True)
    
    if os.path.isfile(filepath+f'{data_n}_{types}.npy'):
        print(f"---> {data_n}_{types}.npy file already existed")
        return
    
    print(f"--> Create {data_n}_{types}.npy start")
    
    path = f"{root_path}/data/crop/{types}/{data_n}/*.tif"
    files = natsort.natsorted(glob(path))
    tif_list=[]
    for i in range(len(files)):
        files[i]=files[i].replace("\\", "/")
        tmp = cv2.imread(files[i], cv2.IMREAD_COLOR)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp = cv2.resize(tmp, (image_size, image_size))
        tmp = tmp / 255.0
        tif_list.append(tmp)

    data=np.array(tif_list)
    np.save(filepath+f'{data_n}_{types}.npy',data)
    print("--> Complete")
    
# 학습 이미지 데이터 생성    
def make_train_image_data(cropsize):
    print("Train image dataset create start")
    
    train_data=pd.read_csv(f"{root_path}/data/modeling/climate_train.csv")
    
    data_n=['Height','Slope','Landuse','population_density','NDVI']
    
    for i in range(len(data_n)):
        print("#"*20)
        crop_train(data_n[i],train_data,cropsize)
        convert_npy(data_n[i],'train')
        
    print("#"*30)    
    print("Train image dataset create complete")
    print("#"*30)
        
def make_test_image_data(cropsize):
    print("Test image dataset create start")
    
    # 강원도 경계
    N = 38.61370931
    E = 129.359995
    S = 37.03353708
    W = 127.0950376

    width = (E-W)/399
    height= (N-S)/277

    width_num,height_num=[], []
    for i in range(400):width_num.append(round(W+width*i,7))
    for i in range(278):height_num.append(round(N-height*i,7))
    
    data_n=['Height','Slope','Landuse','population_density','NDVI']
    
    for i in range(len(data_n)):   
        print("#"*20)
        crop_test(data_n[i],cropsize,width_num,height_num)
        convert_npy(data_n[i],'test')
        
    print("#"*30)
    print("Test image dataset create complete")
    print("#"*30)
    
if __name__ == "__main__":
    root_path = os.getenv("filepath")
    image_size = int(os.getenv('imagesize'))
    make_train_image_data(1)
    make_test_image_data(1)
