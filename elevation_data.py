# -*- coding: utf-8 -*-

import os
from os.path import join
import math
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

from time import time

# 로깅
elv_logger = logging.getLogger("elvation")
elv_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('elvation.log')
file_handler.setFormatter(formatter)
elv_logger.addHandler(file_handler)

# path 설정
ROOT_DIR_PATH = os.getcwd()
TRAIN_FEATHER_PATH = join(ROOT_DIR_PATH, 'train.ftr')
TEST_FEATHER_PATH = join(ROOT_DIR_PATH, 'test.ftr')

ELEVATION_DIR_PATH = join(ROOT_DIR_PATH, 'elevation')
ELV_DICT = {}

def read_feather():
    train_df = pd.read_feather(TRAIN_FEATHER_PATH)
    test_df = pd.read_feather(TEST_FEATHER_PATH)
    return train_df, test_df

def find_elevation_file(lat, long):
    name_lat = str(int(lat)).zfill(2)
    name_long = str(int(long)).zfill(3)
    print(lat, long)
    
    try:
        height_file = join(ELEVATION_DIR_PATH, f'N{name_lat}E{name_long}.hgt')

        size = os.path.getsize(height_file)
        dim = int(math.sqrt(size/2))
    
        assert dim*dim*2 == size, 'Invalid file size'
    
        data = np.fromfile(height_file, np.dtype('>i2'), dim*dim).reshape((dim, dim))
    
        lat_row = int(round((lat - int(lat)) * (dim - 1), 0))
        long_row = int(round((long - int(long)) * (dim - 1), 0))
    
    except:
        elv_logger.error("lat :" + str(lat) + "  long :" + str(long))
        return 0
        
    return data[dim - 1 - lat_row, long_row].astype(int)

def match_elevation(lat, long):
    name_lat = str(int(lat)).zfill(2)
    name_long = str(int(long)).zfill(3)
    print(lat, long)
    
    try:
        dim = 1201
        
        lat_row = int(round((lat - int(lat)) * (dim - 1), 0))
        long_row = int(round((long - int(long)) * (dim - 1), 0))
        
        print(ELV_DICT[f'N{name_lat}E{name_long}'][dim - 1 - lat_row, long_row].astype(int))
        return ELV_DICT[f'N{name_lat}E{name_long}'][dim - 1 - lat_row, long_row].astype(int)
    except:
        return 0

def on_memory_elevation_file():
    for lat in range(60):
        for long in range(90,170):
            name_lat = str(lat).zfill(2)
            name_long = str(long).zfill(3)
            print(name_lat, name_long)
            
            try:
                height_file = join(ELEVATION_DIR_PATH, f'N{name_lat}E{name_long}.hgt')
                
                size = os.path.getsize(height_file)
                dim = int(math.sqrt(size/2))
            
                assert dim*dim*2 == size, 'Invalid file size'
            
                ELV_DICT[f'N{name_lat}E{name_long}'] = np.fromfile(height_file, np.dtype('>i2'), dim*dim).reshape((dim, dim))
            except:
                pass

if __name__ == "__main__":   
    train_df, test_df = read_feather()
    
    on_memory_elevation_file()
#    find_elevation_file(lat, long)

    train_df['elevation'] = np.vectorize(match_elevation)(train_df.lat_GMI, train_df.long_GMI)
    train_df.to_feather(join(ROOT_DIR_PATH, 'elv_train.ftr'))

