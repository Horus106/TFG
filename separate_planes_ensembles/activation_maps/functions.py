import enum
from random import random
from re import S
from xml.etree.ElementInclude import include
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import callbacks, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, ZeroPadding2D, concatenate, Average
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16

from sklearn.model_selection import KFold

from keras.utils.vis_utils import plot_model

from collections import Counter

import scipy.ndimage as sci

import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2 as cv

import random

import ranges_of_age as roa

import sys

from datetime import datetime

def create_CNN(n, width, height, depth, summary = False):
    input_shape = (height, width, depth)

    inputs = Input(shape=input_shape)

    filters = [64, 256, 1024]
    f_size = [5 ,5, 3]

    x = inputs

    for i, f in enumerate(filters):
        x = ZeroPadding2D(padding=2)(x)
        x = Conv2D(f, (f_size[i],f_size[i]), padding="valid", activation='ReLU')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(100, activation='ReLU')(x)
    x = Dense(100, activation='ReLU')(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='ReLU')(x)

    model = Model(inputs, x)

    if summary:
        model.summary()

    return model

def create_CNN_Resnet(n, width, height, depth, trainable = False, summary=False):
    input_shape = (height, width, depth)

    inputs = Input(shape=input_shape)

    resnet = ResNet50(include_top=False, input_tensor=inputs, input_shape=input_shape)
    resnet.layers.pop(0)
    resnet.trainable = trainable

    model = Sequential()
    model.add(inputs)
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(100,activation='ReLU'))
    model.add(Dense(100,activation='ReLU'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation='ReLU'))

    model.layers[0]._name += ('_'+str(n))

    if summary:
        model.summary()

    return model

def get_images_list(dataframe,colormode,augment=True,weights=True):
    list = dataframe.to_numpy()

    image_list = []
    if colormode == 'rgb':
        img_names = ['_panorama_ext_X.png', '_panorama_ext_Y.png', '_panorama_ext_Z.png']
    elif colormode == 'grayscale':
        img_names = ['_panorama_SDM.png', '_panorama_NDM.png', '_panorama_GNDM.png']

    if augment:
        for t in list:
            for n in np.arange(20):
                imgs = []
                for i in np.arange(len(img_names)):
                    img = t[1]+'/'+t[1]+'_'+str(n)+img_names[i]
                    imgs.append(img)
                imgs.append(t[2])
                if weights:
                    imgs.append(t[3])
                image_list.append(imgs)

    else:
        for t in list:
            imgs = []
            for i in np.arange(len(img_names)):
                img = t[1]+'/'+t[1]+'_0'+img_names[i]
                imgs.append(img)
            imgs.append(t[2])
            if weights:
                    imgs.append(t[3])
            image_list.append(imgs)

    return np.array(image_list)

def cached_img_read(img_path, img_shape, colormode, image_cache):
    if img_path not in image_cache.keys():
        image = img_to_array(load_img(img_path, color_mode=colormode, target_size=img_shape, interpolation='bilinear')).astype(np.float32)
        image = image / 255.0
        if colormode == 'rgb':
            image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image_cache[img_path] = image
    
    return image_cache[img_path]

def read_images_gen(images, dir, img_shape, datagen, colormode, image_cache):
    if colormode == "grayscale":
        X = np.zeros((len(images), img_shape[0], img_shape[1], 1))
    else:
        X = np.zeros((len(images), img_shape[0], img_shape[1], 3))

    for i, image_name in enumerate(images):
        image = cached_img_read(os.path.join(dir, image_name), img_shape, colormode, image_cache)
        image = datagen.standardize(image)
        X[i] = image

    return X
        
def image_generator(images, dir, batch_size, datagen, img_shape=(108,108), colormode="rgb", shuffle=True, weights=True):
    image_cache = {}

    while True:
        n_imgs = len(images)
        if shuffle:
            indexs = np.random.permutation(np.arange(n_imgs))
        else:
            indexs = np.arange(n_imgs)
        num_batches = n_imgs // batch_size
        
        for bid in range(num_batches):
            batch_idx = indexs[bid*batch_size:(bid+1)*batch_size]
            batch = [images[i] for i in batch_idx]
            img1 = read_images_gen([b[0] for b in batch], dir, img_shape, datagen, colormode, image_cache)
            img2 = read_images_gen([b[1] for b in batch], dir, img_shape, datagen, colormode, image_cache)
            img3 = read_images_gen([b[2] for b in batch], dir, img_shape, datagen, colormode, image_cache)
            label = np.array([b[3] for b in batch]).astype(np.float32)
            if weights:
                label_weights = np.array([b[4] for b in batch]).astype(np.float32)
                yield ([img1, img2, img3], label, label_weights)
            else:
                yield ([img1, img2, img3], label)

def load_image_normalize_grayscale(images,dir,img_shape,outdir,fit_data = True,list_values = []):
    if fit_data:
        img_mean = []
        img_std = []
        for img in images:
            for i in img[:3]:
                image = img_to_array(load_img(os.path.join(dir,i), color_mode='grayscale', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0
                img_mean.append(np.mean(image))
                img_std.append(np.std(image))


        img_mean = np.mean(img_mean)
        img_std = np.mean(img_std)
        
        print(img_mean, img_std)

    else:
        img_mean = list_values[0]
        img_std = list_values[1]

    img_channel = np.full((img_shape[0],img_shape[1],1), img_mean)
    print(np.shape(img_channel))

    means = []
    stds = []

    for img in images:
        for i in img[:3]:
            image = img_to_array(load_img(os.path.join(dir,i), color_mode='grayscale', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0

            image = (image - img_channel) / img_std

            means.append(np.mean(image))
            stds.append(np.std(image))

            if not os.path.exists(os.path.dirname(os.path.join(outdir,i))):
                os.makedirs(os.path.dirname(os.path.join(outdir,i)))
            cv.imwrite(os.path.join(outdir,i),image)

    print('Media final', np.mean(means), np.mean(stds))
    return [img_mean, img_std]

def load_image_normalize_rgb(images,dir,img_shape,outdir,fit_data = True,list_values = []):
    if fit_data:
        r_mean = []
        g_mean = []
        b_mean = []
        r_std = []
        g_std = []
        b_std = []
        for img in images:
            for i in img[:3]:
                image = img_to_array(load_img(os.path.join(dir,i), color_mode='rgb', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0
                r, g, b = np.split(image,3,axis=2)
                r_mean.append(np.mean(r))
                g_mean.append(np.mean(g))
                b_mean.append(np.mean(b))
                r_std.append(np.std(r))
                g_std.append(np.std(g))
                b_std.append(np.std(b))

        r_mean = np.mean(r_mean)
        r_std = np.mean(r_std)
        g_mean = np.mean(g_mean)
        g_std = np.mean(g_std)
        b_mean = np.mean(b_mean)
        b_std = np.mean(b_std)

        print(r_mean,r_std)
        print(g_mean,g_std)
        print(b_mean,b_std)

    else:
        r_mean = list_values[0]
        r_std = list_values[1]
        g_mean = list_values[2]
        g_std = list_values[3]
        b_mean = list_values[4]
        b_std = list_values[5]

    r_channel = np.full((img_shape[0],img_shape[1],1), r_mean)
    g_channel = np.full((img_shape[0],img_shape[1],1), g_mean)
    b_channel = np.full((img_shape[0],img_shape[1],1), b_mean)
    
    print(np.shape(r_channel),np.shape(g_channel),np.shape(b_channel))

    means = []
    stds = []

    for img in images:
        for i in img[:3]:
            image = img_to_array(load_img(os.path.join(dir,i), color_mode='rgb', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0

            r, g, b = np.split(image,3,axis=2)
            r = (r - r_channel) / r_std
            g = (g - g_channel) / g_std
            b = (b - b_channel) / b_std

            image = np.stack((r,g,b),axis=2)
            image = np.squeeze(image,axis=3)

            means.append(np.mean(image))
            stds.append(np.std(image))

            if not os.path.exists(os.path.dirname(os.path.join(outdir,i))):
                os.makedirs(os.path.dirname(os.path.join(outdir,i)))
            
            cv.imwrite(os.path.join(outdir,i),image)

    print('Media final', np.mean(means), np.mean(stds))
    return [r_mean,r_std,g_mean,g_std,b_mean,b_std]

def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

def bin(age, age_list):
    unique_list = []
    for i in np.arange(np.max(age_list)+1):
        if i not in unique_list:
            unique_list.append(i)

    return unique_list.index(age)

def inverse(x):
    v = np.float32(1/x)
    if v == np.Inf:
        v = np.float32(1)
    return v

def calculate_weights(sample_df):
    age_list = sample_df['Age'].to_list()
    bin_index_per_label = [bin(label,age_list) for label in age_list]
    
    N_ranges = max(bin_index_per_label) + 1
    num_samples_of_bin = dict(Counter(bin_index_per_label))

    emp_label_dist = [num_samples_of_bin.get(i,0) for i in np.arange(N_ranges)]
   
    lds_kernel = cv.getGaussianKernel(5,2).flatten()
    eff_label_dist = sci.convolve1d(np.array(emp_label_dist), weights=lds_kernel, mode='constant')
    
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    
    weights = [inverse(x) for x in eff_num_per_label]

    sample_df['Weight'] = weights

def train_test_split(sample_df, size):
    dic_aux = {}
    sub_df = sample_df
    column = 'Number'
    column_number = 0

    range = np.unique(np.array(sub_df[column].to_list()))

    div = int(len(range) * size)
    if div == 0:
        div = 1

    np.random.shuffle(range)

    for i in range[:div]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Train'

    for i in range[div:]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Test'

    set_list = []

    for i in sample_df[column].values:
        set_list.append(dic_aux[i])
    
    sample_df.insert(0,'Set', set_list, True)

    train_df = sample_df[sample_df["Set"] == 'Train']
    train_df = train_df.drop(columns=["Set"])

    test_df = sample_df[sample_df["Set"] == 'Test']
    test_df = test_df.drop(columns=["Set"])

    return train_df, test_df

def range_of_age(n,ranges):
    for i,r in enumerate(ranges):
        if n >= r[0] and n <= r[1]:
            return i

def precision_by_range(y_true, y_pred, ranges, metric='mae'):

    prec_dic = {}
    for r in np.arange(len(ranges)):
        prec_dic[r] = [[],[],[]]

    for yt, yp in zip(y_true, y_pred):
        r_age = range_of_age(yt, ranges)

        if metric == 'mae':
            dif = abs(yt - yp)
            metric_name = 'MAE'
        elif metric == 'mse':
            dif = (yt - yp) * (yt - yp)
            metric_name = 'MSE'
        prec_dic[r_age][0].append(dif)
        prec_dic[r_age][1].append(yp)
        prec_dic[r_age][2].append(yt)
    
    df_precision = pd.DataFrame()

    prec_dic_filter = {}
    for k in prec_dic.keys():
        if len(np.array(prec_dic[k][1])) != 0:
            prec_dic_filter[k] = prec_dic[k]

    df_precision['Range'] = [ranges[k] for k in prec_dic_filter.keys()]
    df_precision['N values'] = [len(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['T values Mean'] = [np.mean(np.array(prec_dic_filter[k][2])) for k in prec_dic_filter.keys()]
    df_precision['T values std'] = [np.std(np.array(prec_dic_filter[k][2])) for k in prec_dic_filter.keys()]
    df_precision['Values Mean'] = [np.mean(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['Values std'] = [np.std(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['Values 1%'] = [np.percentile(np.array(prec_dic_filter[k][1]),1) for k in prec_dic_filter.keys()]
    df_precision['Values 10%'] = [np.percentile(np.array(prec_dic_filter[k][1]),10) for k in prec_dic_filter.keys()]
    df_precision['Values 25%'] = [np.percentile(np.array(prec_dic_filter[k][1]),25) for k in prec_dic_filter.keys()]
    df_precision['Values 50%'] = [np.percentile(np.array(prec_dic_filter[k][1]),50) for k in prec_dic_filter.keys()]
    df_precision['Values 75%'] = [np.percentile(np.array(prec_dic_filter[k][1]),75) for k in prec_dic_filter.keys()]
    df_precision['Values 99%'] = [np.percentile(np.array(prec_dic_filter[k][1]),99) for k in prec_dic_filter.keys()]
    df_precision[metric_name] = [np.mean(np.array(prec_dic_filter[k][0])) for k in prec_dic_filter.keys()]
    df_precision[metric_name+' Std'] = [np.std(np.array(prec_dic_filter[k][0])) for k in prec_dic_filter.keys()]
    df_precision['Error 1%'] = [np.percentile(np.array(prec_dic_filter[k][0]),1) for k in prec_dic_filter.keys()]
    df_precision['Error 10%'] = [np.percentile(np.array(prec_dic_filter[k][0]),10) for k in prec_dic_filter.keys()]
    df_precision['Error 25%'] = [np.percentile(np.array(prec_dic_filter[k][0]),25) for k in prec_dic_filter.keys()]
    df_precision['Error 50%'] = [np.percentile(np.array(prec_dic_filter[k][0]),50) for k in prec_dic_filter.keys()]
    df_precision['Error 75%'] = [np.percentile(np.array(prec_dic_filter[k][0]),75) for k in prec_dic_filter.keys()]
    df_precision['Error 99%'] = [np.percentile(np.array(prec_dic_filter[k][0]),99) for k in prec_dic_filter.keys()]

    return df_precision

def show_stats(true, pred, metric_range='mae'):
    stats_mae = abs(true-pred)
    stats_mse = (true-pred)*(true-pred)
    measures = [stats_mae, stats_mse]

    df_stats = pd.DataFrame()
    df_stats['Metric'] = ['MAE' ,'MSE']
    df_stats['Mean:'] = [np.mean(m) for m in measures]
    df_stats['Std:'] = [np.std(m) for m in measures]
    df_stats['1% value:'] = [np.percentile(m,1) for m in measures]
    df_stats['10% value:'] = [np.percentile(m,10) for m in measures]
    df_stats['25% value:'] = [np.percentile(m,25) for m in measures]
    df_stats['50% value:'] = [np.median(m) for m in measures]
    df_stats['75% value:'] = [np.percentile(m,75) for m in measures]
    df_stats['99% value:'] = [np.percentile(m,99) for m in measures]
    df_stats['Min value:'] = [np.min(m) for m in measures]
    df_stats['Max value:'] = [np.max(m) for m in measures]

    ranges = [roa.ranges_todd,roa.ranges_5,roa.ranges_3]
    df_precision_complete_list = []
    for range in ranges:
        df_precision = precision_by_range(true, pred, range, metric_range)
        if metric_range == 'mae':
            print('Mean of means:', np.mean(df_precision['MAE'].to_numpy()))
        elif metric_range == 'mse':
            print('Mean of means:', np.mean(df_precision['MSE'].to_numpy()))
        print(df_precision.to_string())
        df_precision_complete_list.append(df_precision)

    
    df_precision_complete = pd.concat(df_precision_complete_list)
    print(df_stats.to_string())

    return df_stats, df_precision_complete