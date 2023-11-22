import os
import urllib3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date

from utils import *
from tabular_dataset import *
from image_dataset import *

import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning) 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from load_variables import load_env
api_key,root_path,db_path,image_size=load_env()

future_loc=pd.read_csv(f"{root_path}/data/future_loc.csv",encoding='cp949') # 현재, 미래 데이터 크롤링 
past_loc=pd.read_csv(f"{root_path}/data/aws_loc_list.csv")  # 과거 데이터 크롤링 
model = tf.keras.models.load_model(f"{root_path}/data/modeling/model")
today = date.today()

inputdate=input("yyyymmdd 형식으로 적어주세요 ")
dates= datetime.strptime(inputdate, "%Y%m%d").date()

# 미래 데이터는 사실 자동으로 30분마다 돌아가는 것을 구상하고 만듦
# 프로젝트가 종료되었지만 정상적으로 작동하며, 입력시간의 130, 230,330 뒤의 산불예보를 가져다줌 -> 즉 input이 현재 시점
# 과거의 경우는 우리가 입력한 시간의 산불예보를 가져다 줌.
if(dates>=today):
    print("미래의 산불발생확률을 확인합니다.")
    tense=1
    times=input("tt30 형식으로 적어주세요")
    filenames=iffuture(inputdate,times,future_loc)
else:
    print("과거의 산불발생확률을 확인합니다.")
    tense=0
    times=input("tt형식으로 적어주세요.")
    filenames=ifpast(inputdate,times,past_loc)

for i in range(len(filenames)):
    print(f"Create {filenames[i]} result")
    if os.path.isfile(f'{db_path}/{filenames[i]}/{filenames[i]}.npy') == False: test_data_interpolation(filenames,tense)
    else: print("Create result start")
    if os.path.isfile(f'{db_path}/{filenames[i]}/{filenames[i]}.png') == False:
        climate_test=np.load(f'{db_path}/{filenames[i]}/{filenames[i]}.npy')
        Height_test=np.load(f'{root_path}/data/modeling/image_data/{image_size}/test/Height_test.npy')
        NDVI_test=np.load(f'{root_path}/data/modeling/image_data/{image_size}/test/NDVI_test.npy')
        Slope_test=np.load(f'{root_path}/data/modeling/image_data/{image_size}/test/Slope_test.npy')
        landuse_test=np.load(f'{root_path}/data/modeling/image_data/{image_size}/test/Landuse_test.npy')
        popden_test=np.load(f'{root_path}/data/modeling/image_data/{image_size}/test/population_density_test.npy')
        x_test = {
            'height_input': Height_test,
            'ndvi_input': NDVI_test,
            'slope_input': Slope_test,
            'landuse_input': landuse_test,
            'popden_input': popden_test,
            'climate_input':climate_test
        }
        y_pred = model.predict(x_test)

        result_arr = np.zeros((278, 400))
        x = 0
        for j in range(278):
            for k in range(400):
                result_arr[j, k] = y_pred[x]  # 결과 배열에 값 추가
                x += 1
                
        cmaps = plt.cm.colors.ListedColormap(['blue', 'yellow','orange', 'red'])  # 파란색, 노란색, 빨간색 순서로 리스트 생성
        bounds = [0, 0.5, 0.7, 0.95, 1]  # 범위 설정
        norm = plt.cm.colors.BoundaryNorm(bounds, cmaps.N)  # 범위와 컬러맵의 개수를 사용하여 정규화 객체 생성

        fig=plt.figure(figsize=(10,7))
        plt.title(f"{inputdate}", fontsize=20)
        plt.imshow(result_arr, cmap=cmaps, norm=norm)
        plt.colorbar()
        fig.savefig(f"{db_path}/{filenames[i]}/{filenames[i]}.png", transparent = True,dpi=300, bbox_inches='tight');
        print("Complete. Check the result on db path.")
    else:
        print("Result is already existed on db path.")