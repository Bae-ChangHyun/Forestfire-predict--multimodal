import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess

from tabular_dataset import *
from image_dataset import *

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Conv1D,Add,Activation
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices, 'GPU')

import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning) 

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    #! GPU 사용 가능
    for gpu in gpus:
        print("GPU:", gpu)
else:
    #! GPU 사용 불가능
    print("GPU not found")
    
from load_variables import load_env
api_key,root_path,db_path,image_size=load_env()
    
subprocess.run(['python', 'tabular_dataset.py'])
subprocess.run(['python', 'image_dataset.py'])

def load_dataset():
    climate_train=pd.read_csv(f'{root_path}/data/modeling/climate_train.csv')
    climate_train.drop(['lon', 'lat'],axis=1,inplace=True)
    x,y=[],[]
    for j in tqdm(range(len(climate_train))):
        x.append(np.array(climate_train.loc[j, ['humidity', 'rainfall', 'temp', 'windspeed']]).astype(float))
        y.append(np.array(climate_train.loc[j, ['target']]).astype(float))
    climate = np.array(x)
    y = np.array(y)
    return climate,y
    
def residual_block(input_layer, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same')(input_layer)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    
    if input_layer.shape[-1] != filters:
        input_layer = Conv2D(filters, (1, 1), padding='same')(input_layer)
    
    x = Add()([input_layer, x])
    x = Activation('relu')(x)
    return x

def res_layers(input_layer):
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(input_layer)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    
    x = residual_block(x, 64, (3, 3))
    x = residual_block(x, 64, (3, 3))
    
    x = residual_block(x, 128, (3, 3))
    x = residual_block(x, 128, (3, 3))
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    return x

def cnn_layers(input_layer):
    conv1_im = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
    pool1_im = MaxPooling2D(pool_size=(2,2))(conv1_im)
    
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(pool1_im)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    
    flatten = Flatten()(pool4)
    
    dense1 = Dense(128, activation='relu')(flatten)
    
    return dense1

def dense_layer(input_layer):
    dense0 = Dense(64, activation='relu')(input_layer)
    
    dense1 = Dense(64, activation='relu')(dense0)
    
    flatten = Flatten()(dense1)
    
    dense2 = Dense(64, activation='relu')(flatten)
    
    return dense2

def construct_model():
    height_cnn = cnn_layers(height_input)
    ndvi_cnn = cnn_layers(ndvi_input)
    slope_cnn = cnn_layers(slope_input)
    landuse_cnn = cnn_layers(landuse_input)
    popden_cnn = cnn_layers(popden_input)
    climate_cnn = dense_layer(climate_input)

    merged = concatenate([height_cnn, ndvi_cnn, slope_cnn, landuse_cnn, popden_cnn, climate_cnn])

    output_layer = Dense(1, activation='sigmoid', name='output_layer')(merged)

    model = Model(inputs=[height_input, ndvi_input,slope_input,landuse_input,popden_input,climate_input], outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    
    return model
    
def train():
    model=construct_model()
    
    x_train = {
    'height_input': height_train,
    'ndvi_input': ndvi_train,
    'slope_input': slope_train,
    'landuse_input': landuse_train,
    'popden_input': popden_train,
    'climate_input':climate
    }
    
    class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.95:
                print("Reached accuracy threshold (0.9). Stopping training.")
                self.model.stop_training = True
    model.fit(x_train, y, epochs=100, batch_size=128, callbacks=[AccuracyThresholdCallback()])
    return model
    
if __name__ == "__main__":
    
    path=f"{root_path}/data/modeling/image_data/{image_size}/train/"
    height_train=np.load(path+'Height_train.npy')
    ndvi_train=np.load(path+'NDVI_train.npy')
    slope_train=np.load(path+'Slope_train.npy')
    landuse_train=np.load(path+'Landuse_train.npy')
    popden_train=np.load(path+'population_density_train.npy')
    climate,y=load_dataset()

    height_input = Input(shape=(image_size, image_size, 3), name='height_input')
    ndvi_input = Input(shape=(image_size, image_size, 3), name='ndvi_input')
    slope_input = Input(shape=(image_size, image_size, 3), name='slope_input')
    landuse_input = Input(shape=(image_size, image_size, 3), name='landuse_input')
    popden_input = Input(shape=(image_size, image_size, 3), name='popden_input')
    climate_input = Input(shape=(4,1), name='climate_input')
    
    model=train()
    model.save(f"{root_path}/data/modeling/model/") 

