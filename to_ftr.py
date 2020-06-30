# -*- coding: utf-8 -*-

import os
from os.path import join
from tqdm.notebook import tqdm
import multiprocessing

import numpy as np
import pandas as pd

from time import time

# data = np.load('./subset_010463_01.npy')

# path
ROOT_DIR_PATH = os.getcwd()
TRAIN_DIR_PATH = join(ROOT_DIR_PATH, 'train')
TEST_DIR_PATH = join(ROOT_DIR_PATH, 'test')
TRAIN_FEATHER_PATH = join(ROOT_DIR_PATH, 'train.ftr')
TEST_FEATHER_PATH = join(ROOT_DIR_PATH, 'test.ftr')

# Column(feature) names
INPUT_COL = [f"temp{i}" for i in range(1, 10)] + ['type', 'long_GMI', 'lat_GMI', 'long_DPR', 'lat_DPR']
TARGET_COL = ['precipitation']
TEST_COL = INPUT_COL
TRAIN_COL = INPUT_COL + TARGET_COL

# Name column (optional)
IS_PAD_NAME_COL = True
NAME_COL = ['orbit', 'subset', 'pixel']
PIXEL_COL = np.arange(1, 1601)[:, None]     # shape (1600,1)

def pad_name_cols(nd, file_name):
    orbit, subset = file_name.split('_')[1:]
    subset = subset[:2]
    nd = np.pad(nd, ((0, 0), (0, 1)), constant_values=int(orbit))
    nd = np.pad(nd, ((0, 0), (0, 1)), constant_values=int(subset))
    return np.c_[nd, PIXEL_COL]

def generate_ndarray_from_file_name(file_name, dir_path):
    file_path = join(dir_path, file_name)
    nd = np.load(file_path).astype(np.float32)  # 40 x 40 x ?
    dim = nd.shape[-1]                          # 차원 확인
    nd = nd.reshape(-1, dim)                    # 1600    x ?
    if IS_PAD_NAME_COL:
        nd = pad_name_cols(nd, file_name)
    return nd

def generate_ndarray_from_dir_path(dir_path):
#    pool = multiprocessing.Pool()
#    nds = pool.starmap(generate_ndarray_from_file_name, [(file_name, dir_path) for file_name in tqdm(os.listdir(dir_path))])
#    
    file_list = [(file_name, dir_path) for file_name in os.listdir(dir_path)]
    nds = []
    count = 0
    for i in file_list:
        nd = generate_ndarray_from_file_name(i[0], i[1])
        nds.append(nd)
        print(str(count) + '/' + str(len(file_list)))
        count+=1
    return np.concatenate(nds)

def generate_dataframe_from_dir_path(dir_path):
    nd = generate_ndarray_from_dir_path(dir_path)
    dim = nd.shape[-1]
    df =  pd.DataFrame(nd,
                       columns=TRAIN_COL if dim == len(TRAIN_COL) else TEST_COL,
                       dtype=np.float32
                      )
    if IS_PAD_NAME_COL:
        df[['orbit', 'subset', 'pixel']] = df[['orbit', 'subset', 'pixel']].astype(np.int32)
        df.sort_values(by=['orbit', 'subset', 'pixel'], ignore_index=True, inplace=True)
    return df

#@elapsed_time
def main():
    global TRAIN_COL, TEST_COL
    TRAIN_COL, TEST_COL = (TRAIN_COL, TEST_COL) if not IS_PAD_NAME_COL else (TRAIN_COL + NAME_COL, TEST_COL + NAME_COL)
    for dir_path in (TRAIN_DIR_PATH, TEST_DIR_PATH):
        df = generate_dataframe_from_dir_path(dir_path)
        if len(df.columns) == len(TRAIN_COL):
            train_df = df
        else:
            test_df = df
    return train_df, test_df

def to_feather(train_df, test_df):
    train_df.to_feather(TRAIN_FEATHER_PATH)
    test_df.to_feather(TEST_FEATHER_PATH)

def read_feather():
    train_df = pd.read_feather(TRAIN_FEATHER_PATH)
    test_df = pd.read_feather(TEST_FEATHER_PATH)
    return train_df, test_df

if __name__ == "__main__":   
    train_df, test_df = read_feather()
    
    to_feather(train_df, test_df)


#a = generate_ndarray_from_file_name('subset_010679_08.npy','C:\\Users\\User\\Documents\\not_for_git\\orbit_project\\train')
#
#file_name = 'subset_010679_08.npy'
#dir_path = 'C:\\Users\\User\\Documents\\not_for_git\\orbit_project\\train'
#file_path = join(dir_path, file_name)
#
#nd = np.load(file_path).astype(np.float32)
#dim = nd.shape[-1]
#
#nd.reshape(-1, dim)
#
#np.arange(1, 1601).shape
#
#PIXEL_COL.shape
#
