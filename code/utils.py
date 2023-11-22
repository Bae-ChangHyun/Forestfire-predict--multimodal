import os
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import rasterio
import tifffile
import geopandas as gpd
from osgeo import gdal
from pyidw import idw
import shutil

import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from load_variables import load_env
api_key,root_path,db_path,image_size=load_env()

def idw_interpoloate(data_n, o_data):
    """
    - 전국의 모든 지역에 대한 기상데이터가 존재하지 않기 때문에,
    asos의 모든 지점으로 부터 얻은 기상데이터를 이용하여, 각 feature별로 데이터를 보간(역거리가중법 / idw보간)
    - 보간시 하나의 feature라도 없으면 전부 drop( 결측치 )
    - 한 지점이라도 살아있으면 보간이 되서 우선 냅둠 -
    # -->어떤 산불은 지점 여러개로 보간된 데이터, 어떤건 지점 한,두개로 보간된 데이터
    """

    print("Start climate interpolation")
    filepath = f"{root_path}/data/data_set({data_n})/interpolate_climate/"
    os.makedirs(filepath, exist_ok=True)

    features = ['humidity', 'wind_sp', 'rainfall', 'temp']

    os.makedirs(filepath+features[0], exist_ok=True)
    os.makedirs(filepath+features[1], exist_ok=True)
    os.makedirs(filepath+features[2], exist_ok=True)
    os.makedirs(filepath+features[3], exist_ok=True)

    weather_data = pd.read_csv(f"{root_path}/data/data_set({data_n})/{data_n}_climate.csv", encoding='cp949')
    weather_data.columns = ['num', 'loc_info', 'lon', 'lat','time', 'humidity', 'wind_sp', 'rainfall', 'temp']
    
    # time이 null이면 데이터가 정상적으로 크롤링되지 않은 것임.
    weather_data.dropna(subset=['time'], inplace=True)
    weather_data['rainfall'] = weather_data['rainfall'].fillna(0)  # 강수가 비어있는건 0으로 채움

    file_count = len(os.listdir(filepath+features[0]))
    if (file_count == len(o_data)):
        print("--> Data is already existed.")
        return
    else:start = file_count-1  # 데이터가 일부만 있을 때 중간부터 시작하기 위해서.

    for i in tqdm(range(start, len(o_data))):
        tmp = weather_data[weather_data['num'] == i]
        tmp = tmp.dropna()
        # 결측치 전부 drop 후에 데이터가 없으면 보간하지 않음
        if (len(tmp) == 0):
            print("There is no data")
            continue
        tmp.drop(['num', 'loc_info', 'time'], axis=1, inplace=True)
        tmp = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp.lon, tmp.lat))
        tmp.to_file(f'{root_path}/data/data{i}.shp')

        for j in range(len(features)):
            idw.idw_interpolation(
                # 보간하고자 하는 shp 파일
                input_point_shapefile=f'{root_path}/data/data{i}.shp',
                # 경계 shp 파일(현재 강원도)
                extent_shapefile=f"{root_path}/data/geo_data/gw_boundary/boundary.shp",
                column_name=features[j],  # 보간하고자 하는 feature 이름.
                power=2,  # 거리 가중치 계수
                search_radious=8,  # 검색하고자 하는 범위
                output_resolution=400,  # 결과물 해상도
            )
            image = rasterio.open(f"{root_path}/data/data{i}_idw.tif")
            image = pd.DataFrame(image.read(1))
            image.to_csv(f"{filepath}{features[j]}/data{i}_idw.csv", encoding='cp949')
            os.remove(f"{root_path}/data/data{i}_idw.tif")

        os.remove(f"{root_path}/data/data{i}.shp")
        os.remove(f"{root_path}/data/data{i}.cpg")
        os.remove(f"{root_path}/data/data{i}.dbf")
        os.remove(f"{root_path}/data/data{i}.shx")
    print("-->Complete")
    print("---------------")
    
def test_data_interpolation(filenames,tense):
    features=['humidity','wind_sp','rainfall','temp']
    print("creating..")
    for i in range(len(filenames)):
        image_list=[]
        data=pd.read_csv(f'{db_path}/{filenames[i]}/{filenames[i]}.csv',encoding='cp949')
        
        if(tense==1):
            data.drop(['baseDate','baseTime','fcstDate','fcstTime'],axis=1,inplace=True)
            data.columns=['nx','ny','humidity','rainfall','temp','wind_sp']
            data['rainfall'] = data['rainfall'].replace('강수없음', '0mm')
            data['rainfall']=[j[:-2] for j in data['rainfall']]
            data['rainfall']=data['rainfall'].fillna(0)
            data['rainfall']=data['rainfall'].astype('float')
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.ny, data.nx)).drop(['nx','ny'],axis=1)
        else:
            data.columns=['loc_info', 'longitude', 'latitude', 'time','humidity', 'wind_sp', 'rainfall','temp']
            data.dropna(subset=['time'], inplace=True)
            data['rainfall']=data['rainfall'].fillna(0)
            data=data.dropna()
            data.drop(['loc_info','time'],axis=1,inplace=True)
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
            
        data.to_file(f'tmp.shp')

        for j in range(len(features)):
            if os.path.isfile(f"{db_path}/{filenames[i]}/{filenames[i]}_{features[j]}.tif") == False:
                idw.idw_interpolation(
                    input_point_shapefile=f'tmp.shp', # 보간하고자 하는 shp 파일 
                    extent_shapefile=f"{root_path}/data/geo_data/gw_boundary/boundary.shp", # 경계 shp 파일(현재 강원도)
                    column_name=features[j], # 보간하고자 하는 feature 이름. 
                    power=2, # 거리 가중치 계수 
                    search_radious=8, # 검색하고자 하는 범위 
                    output_resolution=400, # 결과물 해상도 
                )
                image=rasterio.open(f'tmp_idw.tif')
                image=pd.DataFrame(image.read(1))
                image_list.append(image)
                
                shutil.move(f'tmp_idw.tif', f"{db_path}/{filenames[i]}/interpolation/{filenames[i]}_{features[j]}.tif")
            else:print(f"{features[j]} is existed")
            
        temps,hums,rains,winds=[],[],[],[]
        for j in range(len(image.index)):
            for k in range(len(image.columns)):
                hums.append(image_list[0].iloc[j][k])
                rains.append(image_list[1].iloc[j][k])
                temps.append(image_list[2].iloc[j][k])
                winds.append(image_list[3].iloc[j][k])
                
        climates = {'temp': temps, 'hum': hums, 'rain': rains, 'wind': winds}
        df = pd.DataFrame(climates)
        df=df.replace(32767.0,-9999)
        x_train=[]
        for j in tqdm(range(len(df))):
            x_train.append(np.array(df.loc[j, ['temp','hum','rain','wind']]).astype(float))
        climate = np.array(x_train)
        np.save(f'{db_path}/{filenames[i]}/{filenames[i]}.npy', climate)
        #os.remove(f'{db_path}/{filenames[i]}.csv')
    os.remove(f"tmp.shp")
    os.remove(f"tmp.cpg")
    os.remove(f"tmp.dbf")
    os.remove(f"tmp.shx")
    #os.remove(f"tmp_idw.tif")
    
def find_fireloc(data_n, o_data):
    """
    다른 위치의 기상 데이터는 필요없고, 산불 발생 위치의 기상 데이터만 필요하기 때문에 산불 발생 위치를 찾는 코드.
    
    산불 발생 위치의 경위도는 알고 있지만, 보간된 기상데이터에서 산불 발생 위치의 데이터를 가져오기 위해선,
    아래와 같이 생성한 shp->tif로 변환 후, 강원도 밖은 32767 강원도 내부는 -9999, 산불 발생 위치는 624로 구분.
    
    이후 624 부분을 찾아서 가져옴.(624는 임의의 숫자)
        
    --> 추후,다른 부분의 기상이 필요하지 않다면, 기상청만큼 신뢰성 있는 사이트에서 기상데이터 한개만을 가져오는 방식이 훨씬 효율적
    """

    print("Start find fire loc")
    filepath = f"{root_path}/data/data_set({data_n})/fire_loc/"
    os.makedirs(filepath, exist_ok=True)

    file_count = len(os.listdir(filepath))
    if (file_count == len(o_data)):
        print("--> Data is already existed.")
        return
    else:start = file_count-1  # 데이터가 일부만 있을 때 중간부터 시작하기 위해서.

    if (len(glob(filepath + '/*'))) == 0:
        for i in tqdm(range(start, len(o_data))):
            tmp = pd.DataFrame(o_data.iloc[i])
            tmp = tmp.T
            tmp = gpd.GeoDataFrame(
                tmp, geometry=gpd.points_from_xy(tmp.lon, tmp.lat))
            tmp.drop(['date', 'time', 'lon', 'lat', 'input'],axis=1, inplace=True)
            tmp.to_file(f'{root_path}/data/fire_data.shp')

            shp_path = f"{root_path}/data/fire_data.shp"  # 현재 shp파일 이름
            boundary_path = f'{root_path}/data/geo_data/gw_boundary/boundary.shp'
            tif_path = f"{root_path}/data/fire_data.tif"  # 만들고자 하는 tif파일 이름

            driver = gdal.GetDriverByName('GTiff')
            # 산불 shp 파일을 읽어오기
            shp_datasource = gdal.OpenEx(shp_path, gdal.OF_VECTOR)
            # 경계shp 파일을 읽어오기
            boundary_datasource = gdal.OpenEx(boundary_path, gdal.OF_VECTOR)
            # tif 파일 생성
            tif_datasource = driver.Create(tif_path, 400, 278, 1, gdal.GDT_Float32)
            # 좌표계 설정
            tif_datasource.SetProjection(shp_datasource.GetProjection())
            # 강원도 경계값 가져옴
            boundary = gpd.read_file(boundary_path)
            xmin, ymin, xmax, ymax = boundary.total_bounds
            # 강원도 경계값 가져온걸 위에서 설정한 400,278 즉 액셀 파일 크기형태로 설정
            tif_datasource.SetGeoTransform((xmin, (xmax-xmin)/400, 0, ymax, 0, -(ymax-ymin)/278))
            band = tif_datasource.GetRasterBand(1)
            # 산불 난 지점 외에는 전부 32767으로 설정
            band.Fill(32767)
            band.SetNoDataValue(32767)
            # 강원도 내부는 -9999로 채우기
            gdal.RasterizeLayer(tif_datasource, [1], boundary_datasource.GetLayer(), burn_values=[-9999], options=["ALL_TOUCHED=TRUE"])
            # 산불 발생 위치는 624로 설정
            gdal.RasterizeLayer(tif_datasource, [1], shp_datasource.GetLayer(), burn_values=[624])
            # gdal.RasterizeLayer(tif_datasource, [1], boundary_datasource.GetLayer(), burn_values=[32767])
            shp_datasource = None
            tif_datasource = None
            tif_file = rasterio.open(f"{root_path}/data/fire_data.tif")
            data = tifffile.imread(f"{root_path}/data/fire_data.tif")
            data = pd.DataFrame(data)
            data.to_csv(f"{filepath}fire_data{i}.csv")
            tif_file.close()

        # 필요없는 파일 삭제
        os.remove(f"{root_path}/data/fire_data.shp")
        os.remove(f"{root_path}/data/fire_data.cpg")
        os.remove(f"{root_path}/data/fire_data.dbf")
        os.remove(f"{root_path}/data/fire_data.shx")
        os.remove(f"{root_path}/data/fire_data.tif")
        print("--> Complete")
        print("---------------")
    else:
        print("--> Data is already existed.")
        print("---------------")
        
# Raster 간 해상도 통일 & 해상도 통일 후 테두리 정리(not used)
def image_to_array(InputImage):
    Image = gdal.Open(InputImage, gdal.GA_Update)
    array = Image.ReadAsArray()
    print(array.shape)
    return array

def array_to_image(InputArr, OutputImage, RefImage):
    Image = gdal.Open(RefImage, gdal.GA_Update)
    ImageArr = Image.ReadAsArray()
    
    open(OutputImage, 'w')
    Output = gdal.GetDriverByName('GTiff').Create(OutputImage, ImageArr.shape[1], ImageArr.shape[0], 1, gdal.GDT_Float32)
    Output.GetRasterBand(1).WriteArray(InputArr)
    Output.SetProjection(Image.GetProjection())
    Output.SetGeoTransform(Image.GetGeoTransform())

    Image = None
    Output = None
    
def match_resolution(InputImage, OutputImage, RefImage):
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)
    output_width = Image.RasterXSize
    output_height = Image.RasterYSize

    open(OutputImage, 'w')
    command = ['gdal_translate', '-outsize', str(output_width), str(output_height), '-r', 'bilinear', InputImage, OutputImage]

    subprocess.run(command)
    Output = gdal.Open(OutputImage)
    Output.SetGeoTransform(Image.GetGeoTransform())
    Output.SetProjection(Image.GetProjection())

    Image = None
    Output = None
        
def clear_boundary_using_arr(InputImage, OutputImage, RefImage):
    InputArr = image_to_array(InputImage)
    Image = gdal.Open(RefImage, gdal.GA_Update)
    ImageArr = Image.ReadAsArray()
    
    out_boundary_lat = []
    out_boundary_lon = []
    out_value = ImageArr[0, 0]
    for i in range(Image.RasterYSize):
        for j in range(Image.RasterXSize):
            if ImageArr[i, j] == out_value:
                out_boundary_lat.append(i)
                out_boundary_lon.append(j)
    InputArr[out_boundary_lat, out_boundary_lon] = InputArr[0, 0]
    array_to_image(InputArr, OutputImage, RefImage)
    
