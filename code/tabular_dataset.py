from utils import *
import os
import time
import json
import random
import requests
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import geopandas as gpd
from dotenv import load_dotenv
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from load_variables import load_env
api_key,root_path,db_path,image_size=load_env()

# Crawling hourly asos weather data
def asos_crawling(date, locn):
    # crawling url
    url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
    # input date preprocessing
    startDt = datetime.strptime(date[:8], '%Y%m%d')  # start date
    startHh = datetime.strptime(date[8:], '%H')   # start time
    endHh = (startHh + timedelta(hours=1)) # start time + 1 (using timedelta for date change)
    if (endHh.strftime('%H:%M:%S').split(':')[0] == '00'):endDt = startDt + timedelta(days=1)
    else:endDt = startDt  # endHh==00이면 시작날짜와 종료날짜가 달라야함. 22일 23시와 23일 00시 이런식.

    # input format
    startDt = startDt.strftime('%Y%m%d')
    startHh = startHh.strftime('%H')
    endDt = endDt.strftime('%Y%m%d')
    endHh = endHh.strftime('%H')

    params = {'serviceKey': api_key,
              'pageNo': '1',
              'numOfRows': '10',
              'dataType': 'JSON',
              'dataCd': 'ASOS',
              'dateCd': 'HR',
              'startDt': startDt,  # startdate
              'startHh': startHh,  # starttime
              'endDt': endDt,  # end date
              'endHh': endHh,  # end time
              'stnIds': locn
              }
    
    max_retries = 5
    for retry in range(max_retries):
        try:
            # https://www.data.go.kr/data/15057210/openapi.do
            response = requests.get('http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList', params=params, verify=False)
            response.raise_for_status()  # HTTP error check
            json_obj = json.loads(response.content)
            item = json_obj.get('response', {}).get('body', {}).get('items', {}).get('item', [{}])[0]
            if item:
                times = item.get('tm')
                humidity = item.get('hm')
                windspeed = item.get('ws')
                rain = item.get('rn')
                temp = item.get('ta')
                return times, humidity, windspeed, rain, temp
            else:
                print(f'No data found')
                time.sleep(5)
        except:
            print(f'An error occurred. Retying. Don\'t quit')
            time.sleep(2)
    return np.nan, np.nan, np.nan, np.nan, np.nan

def future_weather_crawling(date,times,locn):
    url = 'https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'
    
    params ={'serviceKey' : api_key,
             'pageNo' : '1',
             'numOfRows' : '1000',
             'dataType' : 'JSON', 
             'base_date' : date, 
             'base_time' : times, 
             'nx' : locn[1], 
             'ny' : locn[0], 
            }
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(url, params=params,verify=False)
            try:
                json_obj = json.loads(response.content)
                try:
                    json_obj=json_obj["response"]["body"]["items"]["item"]       
                    return json_obj
                except:
                    print("Retry1")
                    print(json_obj)
                    time.sleep(5)
            except:
                print(response.content)
                try:
                    json_obj = json.loads(response.content)
                    print(response,json_obj)
                    print("Retry2")
                    time.sleep(5)
                except json.JSONDecodeError:
                    time.sleep(2)
                    continue  
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            print(f'Network ERROR')
            time.sleep(2)
    return np.nan 

def collect_climate(data_n, o_data, method):
    o_data['input'] = o_data['input'].astype(str)
    print("Start climate data collect")
    data_dir = f"{root_path}/data/data_set({data_n})/climate_data/"
    os.makedirs(data_dir, exist_ok=True)
    loc_data = pd.read_csv(f"{root_path}/data/aws_loc_list.csv")
    loc_list = loc_data['지점번호']
    result = []

    file_count = len(os.listdir(data_dir))
    if (file_count == len(o_data)):
        print("--> Data is already existed.")
        return
    else:
        start = file_count
        print(f"Continuing collect data at {start}")

    # this method is using downloaded asos csv file (pre download required)
    if method == 1:
        for i in tqdm(range(start-1,len(o_data))):
            tmp = pd.DataFrame(columns=['num', 'loc_info', 'lon', 'lat', 'time', 'humidity', 'wind_sp', 'rainfall', 'temp'])
            for j in range(len(loc_list)):
                new_data = [i, loc_list[j], loc_data['lon'][j], loc_data['lat'][j]]
                new_data.extend(asos_crawling(o_data['input'][i], loc_list[j]))
                tmp = pd.concat([tmp, pd.DataFrame([new_data], columns=tmp.columns)], ignore_index=True)
            tmp.to_csv(f"{data_dir}data_{i}.csv", encoding='cp949', index=False)
    # this method is crawling data ny using api
    elif method == 2:
        for i in tqdm(range(start-1, len(o_data))):
            year = o_data['input'][i][:4]
            df_asos_hr = pd.read_csv(f'{root_path}/data/ASOS_Hr/ASOS_Hr_{year}.csv', low_memory=False)
            date = datetime.strptime(o_data['input'][i], "%Y%m%d%H")
            date = date.strftime("%Y-%m-%d %H:%M")

            df_asos_hr = df_asos_hr[df_asos_hr['일시']== date].reset_index(drop=True)
            df_asos_hr['num'] = i
            df_asos_hr['longitude'] = df_asos_hr.apply(lambda row: loc_data.loc[loc_data['지점번호'] == row['지점'], 'lon'].values[0]
                                                       if loc_data.loc[loc_data['지점번호'] == row['지점'], 'lon'].shape[0] != 0 else np.nan, axis=1)
            df_asos_hr['latitude'] = df_asos_hr.apply(lambda row: loc_data.loc[loc_data['지점번호'] == row['지점'], 'lat'].values[0]
                                                      if loc_data.loc[loc_data['지점번호'] == row['지점'], 'lat'].shape[0] != 0 else np.nan, axis=1)

            df_asos_hr = df_asos_hr[['num', '지점', 'longitude', 'latitude', '일시', '습도(%)', '풍속(m/s)', '강수량(mm)', '기온(°C)']]
            df_asos_hr.columns = ['num', 'loc_info', 'lon', 'lat','time', 'humidity', 'wind_sp', 'rainfall', 'temp']
            df_asos_hr.to_csv(f'{data_dir}/data_{i}.csv', encoding='cp949', index=False)
            tmp = pd.read_csv(f"{data_dir}data_{i}.csv", encoding='cp949')
    weather_data = pd.concat(map(pd.read_csv, glob(f"{root_path}/data/data_set({data_n})/climate_data/*.csv")), ignore_index=True)       
    weather_data.to_csv(f"{root_path}/data/data_set({data_n})/{data_n}_climate.csv",encoding='cp949', index=False)
    print("Complete")

def get_climate(data_n, o_data):
    """
    - 보간한 기상데이터에서 산불발생위치의 기상 데이터만을 가져옴.
    """
    print("Start get climate information")
    filepath = f"{root_path}/data/data_set({data_n})/train_{data_n}.csv"

    if os.path.isfile(filepath):
        print("--> Data is already existed.")
        print("---------------")
    else:
        climate = []
        for i in tqdm(range(len(o_data))):
            data = pd.read_csv(f"{root_path}/data/data_set({data_n})/fire_loc/fire_data{i}.csv")
            index = data[data == 624].stack().index[0] 
            temp = pd.read_csv(f"{root_path}/data/data_set({data_n})/interpolate_climate/temp/data{i}_idw.csv")
            rain = pd.read_csv(f"{root_path}/data/data_set({data_n})/interpolate_climate/rainfall/data{i}_idw.csv")
            hums = pd.read_csv(f"{root_path}/data/data_set({data_n})/interpolate_climate/humidity/data{i}_idw.csv")
            wind = pd.read_csv(f"{root_path}/data/data_set({data_n})/interpolate_climate/wind_sp/data{i}_idw.csv")

            r, c = index[0], int(index[1])
            tmp_data = [
                [a, b, c, d]
                for a, b, c, d in zip(
                    [temp.iloc[r][c]],
                    [rain.iloc[r][c]],
                    [hums.iloc[r][c]],
                    [wind.iloc[r][c]]
                )
            ]
            climate.append(tmp_data[0])
        climate = pd.DataFrame(climate, columns=['기온', '강수', '습도', '풍속'])
        o_data.drop('input', axis=1, inplace=True)
        trainfire = pd.concat([o_data, climate], axis=1, ignore_index=True)
        trainfire.columns = ['date', 'time', 'lon', 'lat','temp', 'rainfall', 'humidity', 'windspeed']
        trainfire.to_csv(filepath, index=False)
        print("--> Complete")
        print("---------------")

def make_random_dataset(start_year, end_year, num_samples, filepath):
    """
    - 산불 난 데이터는 target==1로 실존 데이터이고, 산불 나지 않은 데이터는 target==0으로 임의로 만든 데이터 셋이다.
    - 데이터 셋을 만들 떄, 어떻게 만드냐에 따라 매우 달라진다.
    - 같은 날씨에 같은 지형이 들어가는 경우는 없어야 하며, 계절과 날짜 모두 랜덤으로 골고루게 되었다.
    - 데이터는 반드시 강원도 이내에 있어야 한다.
    - 현재 만든 방식은 
      1. 같은 지점을 총 5번씩 다른 날짜에 뽑았음--> 228*5=1140개의 데이터셋 만듦 
    """
    seasons = ['spring', 'fall', 'winter', 'else']
    # set data ratio-> spring: 0.6, summer: 0.2, autumn: 0.1, winter: 0.1
    season_weights = [0.6, 0.2, 0.1, 0.1]

    min_latitude, max_latitude = 37.03353708, 38.61370931
    min_longitude, max_longitude = 127.0950376, 129.359995

    tmp = []
    for _ in range(num_samples):
        year = random.randint(start_year, end_year)
        hour = random.randint(0, 23)
        season = random.choices(seasons, weights=season_weights)[0]

        if season == 'spring':month = random.randint(2, 5)
        elif season == 'fall':month = random.randint(11, 12)
        elif season == 'winter':month = random.randint(6, 10)
        else:month = random.randint(1, 1)

        day = random.randint(1, 28)
        minute, second = 0, 0
        time = pd.Timestamp(year, month, day, hour, minute, second)

        tmp.append(time)

    dates_df = pd.DataFrame(tmp, columns=['Date'])
    dates_df['date'] = pd.to_datetime(dates_df['Date']).dt.strftime('%Y%m%d')
    dates_df['time'] = pd.to_datetime(dates_df['Date']).dt.strftime('%H%M%S')

    dates_df["date"] = dates_df["date"].str.zfill(8)
    dates_df["time"] = dates_df["time"].str.zfill(6)

    dates_df.drop('Date', axis=1, inplace=True)
    tmp = []
    for _ in range(500):  # 강원도 바깥에생성될 걸 대비핵서 넉넉하게
        while True:
            latitude = random.uniform(min_latitude, max_latitude)
            longitude = random.uniform(min_longitude, max_longitude)
            coordinate = (latitude, longitude)
            if coordinate not in tmp:
                tmp.append((longitude, latitude))
                break
    df = pd.DataFrame(tmp, columns=['lon', 'lat'])
    # plt.scatter(df.lon,df.lat) # check data distribution(데이터 분포 확인)

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    df.crs = "EPSG:4326"
    df.to_file(f'df.shp')
    df = gpd.read_file('df.shp')
    gangwon_df = gpd.read_file(f"{root_path}/data/geo_data/gw_boundary/boundary.shp")
    gangwon_boundary = gangwon_df.geometry.unary_union
    df = df[df.geometry.within(gangwon_boundary)][:228]
    df.reset_index(drop=True, inplace=True)
    os.remove('df.shp')
    df.drop('geometry', axis=1, inplace=True)
    df = pd.concat([df, df, df, df, df], axis=0, ignore_index=True)

    notfire = pd.concat([dates_df, df], axis=1)

    notfire['input'] = notfire['date'].astype(str)+notfire['time'].apply(lambda x: str(x)[:2]).str.zfill(2).astype(str)
    notfire.to_csv(filepath, index=False)
    

def iffuture(date,times,loc_list):
    result_list=[]
    fulldate=date+times
    
    file_name = datetime.strptime(fulldate, "%Y%m%d%H%M")
    file_1 = (file_name + timedelta(hours=1, minutes=30)).strftime("%Y%m%d%H%M")
    file_2 = (file_name + timedelta(hours=2, minutes=30)).strftime("%Y%m%d%H%M")
    file_3 = (file_name + timedelta(hours=3, minutes=30)).strftime("%Y%m%d%H%M")
    
    os.makedirs(f"{db_path}/{file_1}/interpolation", exist_ok=True)
    os.makedirs(f"{db_path}/{file_2}/interpolation", exist_ok=True)
    os.makedirs(f"{db_path}/{file_3}/interpolation", exist_ok=True)
    
    for i in tqdm(range(len(loc_list))):
        loc=(loc_list['격자 Y'].iloc[i],loc_list['격자 X'].iloc[i])

        result=pd.DataFrame(future_weather_crawling(date,times,loc))

        t_1=file_1[-4:]
        t_2=file_2[-4:]
        t_3=file_3[-4:]

        time_list = [t_1,t_2,t_3]

        output = result[result['fcstTime'].isin(time_list)]
        output = output.pivot(index=['baseDate', 'baseTime', 'fcstDate', 'fcstTime', 'nx', 'ny'], columns='category', values='fcstValue').reset_index()

        output.drop(['LGT','VEC','SKY','UUU','VVV','PTY'],axis=1,inplace=True)
        output.columns=['baseDate', 'baseTime', 'fcstDate', 'fcstTime', 'nx', 'ny','습도','강수량','기온','풍속']
        result_list.append(output)

    test = pd.concat(result_list, ignore_index=True)

    tmp1=test[test['fcstTime']==t_1].reset_index(drop=True)
    tmp2=test[test['fcstTime']==t_2].reset_index(drop=True)
    tmp3=test[test['fcstTime']==t_3].reset_index(drop=True)

    tmp1.to_csv(f'{db_path}/{file_1}/{file_1}.csv',index=False,encoding='cp949')
    tmp2.to_csv(f'{db_path}/{file_2}/{file_2}.csv',index=False,encoding='cp949')
    tmp3.to_csv(f'{db_path}/{file_3}/{file_3}.csv',index=False,encoding='cp949')
    
    return [file_1,file_2,file_3]
    
def ifpast(date,times,loc_list):
    fulldate=date+times
    loc_num=loc_list['지점번호']
    file=fulldate+'00'
    os.makedirs(f"{db_path}/{file}/interpolation", exist_ok=True)
    if os.path.isfile(f"{db_path}/{file}/{file}.csv") == False:
        tmp=pd.DataFrame(columns=['location information', 'longitude','latitude','time','humidity','wind speed','precipitation','temperature'])
        for j in tqdm(range(len(loc_num))):
            result=[loc_num[j],loc_list['lon'][j],loc_list['lat'][j]]
            result.extend(asos_crawling(fulldate,loc_num[j])) 
            tmp = pd.concat([tmp, pd.DataFrame([result], columns=tmp.columns)], ignore_index=True)
            tmp.to_csv(f"{db_path}/{file}/{file}.csv",index=False,encoding='cp949')
    else: print(f"{file} Climate data is already existed. \n skip to next step.")
    
    return [file]

def main(data_n):
    filepath = f"{root_path}/data/gangwon_{data_n}.csv"
    if os.path.isfile(filepath) == False:
        # nofire 데이터가 없을 경우 생성해야함.
        print("Create Random nofire dataset")
        make_random_dataset(2011, 2022, 1140, filepath)
    o_data = pd.read_csv(f"{root_path}/data/gangwon_{data_n}.csv")
    print("#"*50)
    print(f"Make {data_n} dataset")
    collect_climate(data_n, o_data, 2)
    idw_interpoloate(data_n, o_data)
    find_fireloc(data_n, o_data)
    get_climate(data_n, o_data)
    dataset = pd.read_csv(f"{root_path}/data/data_set({data_n})/train_{data_n}.csv")
    return dataset

if __name__ == "__main__":
    fire_dataset=main("fire")
    fire_dataset['target']=1
    nofire_dataset=main("nofire")
    nofire_dataset['target']=0
    final_dataset=pd.concat([fire_dataset,nofire_dataset],axis=0,ignore_index=True)
    #final_dataset.drop(['lon', 'lat'],axis=1,inplace=True)
    final_dataset=final_dataset.replace(32767.0,-9999)
    os.makedirs(f"{root_path}/data/modeling/", exist_ok=True)
    final_dataset.to_csv(f"{root_path}/data/modeling/climate_train.csv",index=False)
