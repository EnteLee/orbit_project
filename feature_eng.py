# -*- coding: utf-8 -*-

import os
from os.path import join
import math

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR_PATH = os.getcwd()
TRAIN_FEATHER_PATH = join(ROOT_DIR_PATH, 'train.ftr')
TEST_FEATHER_PATH = join(ROOT_DIR_PATH, 'test.ftr')

def read_feather():
    train_df = pd.read_feather(TRAIN_FEATHER_PATH)
    test_df = pd.read_feather(TEST_FEATHER_PATH)
    return train_df, test_df

train_df, _ = read_feather()

train_df.head(10)

train_df.iloc[0]['lat_GMI'], train_df.iloc[0]['long_GMI']

train_df.iloc[401]


# 41044800 -> subset1
# 81360000 -> subset2
# 122152000 -> subset3
def gmi_tuple(idx):
    return train_df.iloc[idx]['lat_GMI'], train_df.iloc[idx]['long_GMI']

gmi_tuple(0)
gmi_tuple(399)

x=[]
y=[]
for i in range(len(test_df)):
    if i % 1600 == 0:
        x.append(test_df.iloc[i]['long_GMI'])
        y.append(test_df.iloc[i]['lat_GMI'])
        
plt.scatter(x,y)

min(x),max(x)
min(y),max(y)

# type 빈도 분석
train_df.groupby('type').count()


temp1 = np.load('./train/subset_010462_01.npy') 

train_temp = train_df.head(10)

train_corr = train_df.corr()
train_corr2 = train_df[['temp1','temp2','temp3','temp4','temp5','temp6','temp7','temp8', 'temp9', 'type', 'long_GMI','lat_GMI', 'long_DPR', 'lat_DPR','precipitation']].corr()

plt.figure(figsize=(50,50))

sns.heatmap(data = train_corr2, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')

# 10.65H, 10.65V, 18.7H, 18.7V, 23.8V, 36.5H, 36.5V, 89H, 89V

# 10GHz 채널 -> 열대지방의 강한 강수량 확인을 위함
# 18.7 채널 -> 일반, light 강수량 확인용
# 23.8 -> 다른 채널과 함께 사용시 수증기 흡수를 위함. only Vertical
# 36.5 -> 19채널과 같이 씀. 일반, light 강수량, 일반 강수량 확인시 중요.
# 89 -> multi channel 검색시 사용.
# 165.5 -> 열대지방 밖의 light 강수, (89, 183이랑 같이)
# 183 -> 얼음 산란 신호. 눈덮인 땅(좀 추운곳 위주). only Vertical

# 태풍 데이터 찾기
# 시간대 역추산

# 한국 경도(124~132), 위도(33~43)
korea_df = train_df.query('long_GMI >= 124 and long_GMI <= 132 and lat_GMI >= 33 and lat_GMI <= 43 and pixel < 41044800')

korea_df = korea_df.reset_index(drop=False)

plt.scatter(korea_df.index, korea_df['temp9'])

# 위경도에 따른 해발고도 feature 추가
train_df.columns

korea_df[['long_GMI','lat_GMI']].iloc[0]


height_file = join(ROOT_DIR_PATH, 'elevation\\N36E126.hgt')

size = os.path.getsize(height_file)
dim = int(math.sqrt(size/2))

assert dim*dim*2 == size, 'Invalid file size'

data = np.fromfile(height_file, np.dtype('>i2'), dim*dim).reshape((dim, dim))


# feature importance
model = lgb.LGBMRegressor()

train_df = train_df.drop(['long_DPR','lat_DPR','pixel','subset'], axis=1)

target = 'precipitation'
col_list = train_df.columns.to_list()
col_list.remove(target)

kfold = KFold(n_splits = 5, random_state = 92)

params = {
        "metric": 'MAE'
        }

for fold_, (train_index, val_index) in enumerate(kfold.split(train_df)):
    train_X, val_X = train_df.loc[train_index, col_list],\
                    train_df.loc[val_index, col_list]
    train_y, val_y = train_df.loc[train_index, target], train_df.loc[val_index, target]
    
    train_set = lgb.Dataset(train_X, label = train_y, categorical_feature = ['type','orbit'])
    val_set = lgb.Dataset(val_X, label = val_y, categorical_feature = ['type','orbit'])
    
    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval = 100, early_stopping_rounds = 200)

col_list
imp = model.feature_importance()

plt.bar(col_list, imp)

